"""
TRM SVFT adapter - combines SVFT (Singular Value Fine-Tuning) with TRM recursion.

SVFT decomposes weights via SVD: W = U @ S @ V^T
- U, V are frozen singular vectors (orthonormal bases)
- S is diagonal singular values (frozen as s0)
- sd is sparse learnable delta to S (controlled by gate)
- TRM recursively refines sd in the singular value space

This is similar to DeLoRA but uses sparse singular value updates instead of low-rank deltas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from jaxtyping import Float
from einops import repeat, rearrange

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.config import PeftConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType

try:
    from torch_sparse import SparseTensor
    HAS_TORCH_SPARSE = True
except ImportError:
    HAS_TORCH_SPARSE = False
    print("Warning: torch_sparse not available, SVFT will use dense operations (slower)")

from .trm_adapter import L_net, trm_recursion


@dataclass
class TRMSvftConfig(PeftConfig):
    """
    Configuration for TRM SVFT adapter.
    """
    # SVFT-specific parameters
    r: int = field(default=None, metadata={"help": "SVD rank (None = full rank)"})
    off_diag: int = field(default=1, metadata={"help": "Number of off-diagonals in sparse S matrix"})
    pattern: Literal["banded", "random", "top_k"] = field(
        default="banded", 
        metadata={"help": "Sparsity pattern for sd matrix"}
    )
    fill_orthonormal: bool = field(
        default=False, 
        metadata={"help": "Fill remaining rank with random orthonormal basis"}
    )
    
    # TRM-specific parameters
    l_cycles: int = field(default=6, metadata={"help": "Number of L_net cycles per H cycle"})
    h_cycles: int = field(default=2, metadata={"help": "Number of H cycles"})
    expansion: float = field(default=2.67, metadata={"help": "TRM expansion factor"})
    l_layers: int = field(default=2, metadata={"help": "Number of L_net layers"})
    num_heads: int = field(default=8, metadata={"help": "Number of attention heads"})
    
    # Standard PEFT parameters
    target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of module names to apply adapter to"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules to save (not adapt)"}
    )

    def __post_init__(self):
        self.peft_type = 'TRMSVFT'
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class TRMSvftLayer(BaseTunerLayer):
    """
    TRM SVFT layer that wraps a base layer and applies TRM-enhanced SVFT.
    
    SVFT decomposes W = U @ S @ V^T where:
    - U, V are frozen orthonormal bases from SVD
    - S = s0 + sd where s0 is frozen diagonal, sd is sparse learnable delta
    - TRM recursively refines sd in singular value space
    """
    
    adapter_layer_names = ("svft_u", "svft_v", "svft_s0", "svft_sd", "svft_gate", "svft_l_nets")
    other_param_names = ("r", "off_diag", "pattern", "svft_zL_init", "svft_zH_init")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        BaseTunerLayer.__init__(self, base_layer, **kwargs)
        
        # SVFT components (per adapter)
        self.svft_u = nn.ParameterDict({})
        self.svft_v = nn.ParameterDict({})
        self.svft_s0 = BufferDict({})
        self.svft_sd = nn.ParameterDict({})
        self.svft_gate = nn.ParameterDict({})
        
        # Sparse indices (buffers)
        self.svft_s0_row = BufferDict({})
        self.svft_s0_col = BufferDict({})
        self.svft_sd_row = BufferDict({})
        self.svft_sd_col = BufferDict({})
        
        # TRM components
        self.svft_zL_init = BufferDict({})
        self.svft_zH_init = BufferDict({})
        self.svft_configs: Dict[str, TRMSvftConfig] = {}
        self.svft_l_nets = nn.ModuleDict({})
        
        # Marker for Coconut to find TRM layers
        self._recursion_cache = None
        
        self._active_adapter = None

    def update_layer(
        self,
        adapter_name: str,
        svft_config: TRMSvftConfig,
        **kwargs
    ) -> None:
        """Initialize SVFT adapter on this layer."""
        if adapter_name in self.svft_u:
            return  # Already initialized
        
        self.svft_configs[adapter_name] = svft_config
        
        # Compute SVD of base weight
        base_weight = self.get_base_layer().weight  # [out, in]
        device = base_weight.device
        
        U, S, Vh = torch.linalg.svd(base_weight, full_matrices=False)  # U: [out, min], S: [min], Vh: [min, in]
        
        # Determine rank
        r = S.shape[0] if svft_config.r is None else min(S.shape[0], svft_config.r)
        
        # Optionally fill with orthonormal basis
        if svft_config.fill_orthonormal and r < S.shape[0]:
            diff_rank = S.shape[0] - r
            Q_u = torch.randn(U.shape[0], diff_rank, device=device)
            nn.init.orthogonal_(Q_u)
            Q_v = torch.randn(diff_rank, Vh.shape[1], device=device)
            nn.init.orthogonal_(Q_v)
            
            U = torch.cat([U[:, :r], Q_u], dim=1)
            Vh = torch.cat([Vh[:r, :], Q_v], dim=0)
            S = torch.cat([S[:r], torch.zeros(diff_rank, device=device)], dim=0)
            r = S.shape[0]
        else:
            U = U[:, :r]
            Vh = Vh[:r, :]
            S = S[:r]
        
        # Store frozen U, V, s0
        self.svft_u[adapter_name] = nn.Parameter(U.clone().detach().contiguous(), requires_grad=False)
        self.svft_v[adapter_name] = nn.Parameter(Vh.clone().detach().contiguous(), requires_grad=False)
        self.svft_s0[adapter_name] = S.clone().detach().contiguous()
        
        # Create sparse indices for s0 (diagonal)
        s0_indices = torch.sparse.spdiags(S, torch.LongTensor([0]), (r, r)).coalesce().indices()
        self.svft_s0_row[adapter_name] = s0_indices[0]
        self.svft_s0_col[adapter_name] = s0_indices[1]
        
        # Create sparse pattern for sd
        pattern = svft_config.pattern
        off_diag = svft_config.off_diag
        
        if pattern == "random":
            k = r * (2 * off_diag + 1) - off_diag * (off_diag + 1)
            rows = torch.randint(0, r, (k,), device=device)
            cols = torch.randint(0, r, (k,), device=device)
            sd_indices = torch.stack([rows, cols])
        elif pattern == "banded":
            diags = 2 * off_diag + 1
            offsets_positive = torch.arange(0, off_diag + 1)
            offsets_negative = torch.arange(-1, -off_diag - 1, -1)
            offsets = torch.cat([offsets_positive, offsets_negative])
            sd_indices = torch.sparse.spdiags(
                torch.randn([diags, r]), offsets, (r, r)
            ).coalesce().indices()
            k = sd_indices.shape[1]
        elif pattern == "top_k":
            if U.shape == Vh.shape:
                coeffs = U @ Vh.T
            else:
                coeffs = U if U.shape[0] == U.shape[1] else Vh
            
            k = r * (2 * off_diag + 1) - off_diag * (off_diag + 1)
            flattened = coeffs.contiguous().view(-1)
            _, top_indices_flat = torch.topk(flattened.abs(), k)
            num_rows, num_cols = coeffs.size()
            rows = top_indices_flat // num_cols
            cols = top_indices_flat % num_cols
            sd_indices = torch.stack([rows, cols])
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        self.svft_sd_row[adapter_name] = sd_indices[0]
        self.svft_sd_col[adapter_name] = sd_indices[1]
        
        # Initialize learnable parameters
        sd = torch.zeros(k, device=device)
        nn.init.kaiming_normal_(sd[None, :])
        self.svft_sd[adapter_name] = nn.Parameter(sd, requires_grad=True)
        self.svft_gate[adapter_name] = nn.Parameter(torch.tensor([0.0], device=device), requires_grad=True)
        
        # Initialize TRM components
        self.svft_l_nets[adapter_name] = L_net(
            r,
            svft_config.l_layers,
            svft_config.num_heads,
            svft_config.expansion,
        )
        
        # Initialize TRM recursion states in r_dim
        zH = torch.empty(r, device=device)
        zL = torch.empty(r, device=device)
        torch.nn.init.trunc_normal_(zH, std=1.0)
        torch.nn.init.trunc_normal_(zL, std=1.0)
        self.svft_zL_init[adapter_name] = zL
        self.svft_zH_init[adapter_name] = zH

    def trm(self, adapter_name: str, zL, zH, context_hs, h_cycles=None):
        """Wrapper around trm_recursion with adapter-specific config."""
        svft_config = self.svft_configs[adapter_name]
        if h_cycles is None:
            h_cycles = svft_config.h_cycles
        
        return trm_recursion(
            l_net=self.svft_l_nets[adapter_name],
            zL=zL,
            zH=zH,
            context=context_hs,
            l_cycles=svft_config.l_cycles,
            h_cycles=h_cycles,
        )

    def get_sparse_s_eff(self, adapter_name: str, sd_values: Optional[Tensor] = None) -> Tensor:
        """
        Compute effective singular value matrix: s_eff = s0 + sd
        Returns sparse or dense tensor depending on torch_sparse availability.
        """
        if sd_values is None:
            sd_values = self.svft_sd[adapter_name] * F.sigmoid(self.svft_gate[adapter_name])
        
        r = self.svft_s0[adapter_name].shape[0]
        device = self.svft_s0[adapter_name].device
        
        if HAS_TORCH_SPARSE:
            sd = SparseTensor(
                row=self.svft_sd_row[adapter_name],
                col=self.svft_sd_col[adapter_name],
                value=sd_values
            )
            s0 = SparseTensor(
                row=self.svft_s0_row[adapter_name],
                col=self.svft_s0_col[adapter_name],
                value=self.svft_s0[adapter_name]
            )
            s_eff = s0 + sd
        else:
            # Fallback to dense (slower)
            s0_dense = torch.diag(self.svft_s0[adapter_name])
            sd_dense = torch.zeros(r, r, device=device)
            sd_dense[self.svft_sd_row[adapter_name], self.svft_sd_col[adapter_name]] = sd_values
            s_eff = s0_dense + sd_dense
        
        return s_eff

    def forward(self, x: Float[Tensor, 'b s h'], *args: Any, **kwargs: Any) -> Float[Tensor, 'b s h']:
        previous_dtype = x.dtype
        
        # Use injected cache from Coconut.recursion_context() if available
        assert len(self.active_adapters) <= 1, "TRM SVFT currently supports only one active adapter at a time."
        if self._recursion_cache is None:
            recursion_cache = {}
        else:
            recursion_cache = self._recursion_cache

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(x, *args, **kwargs).to(previous_dtype)

            base_out = self.base_layer(x, *args, **kwargs)
            add_out = torch.zeros_like(base_out)

            for adapter in self.active_adapters:
                if adapter not in self.svft_u:
                    continue

                U = self.svft_u[adapter]
                V = self.svft_v[adapter]
                
                # Check if we're in steering mode (post-latent)
                steering_mode = recursion_cache.get('steering_mode', False)
                
                if steering_mode:
                    # Don't run TRM, just apply cached refined sd
                    zH = recursion_cache.get('zH')
                    if zH is not None:
                        # zH represents refined singular value deltas
                        sd_values = zH * F.sigmoid(self.svft_gate[adapter])
                        s_eff = self.get_sparse_s_eff(adapter, sd_values.detach())
                        
                        # Apply: x @ V^T @ s_eff @ U^T
                        if self.training:
                            h = (x @ V.T) @ s_eff.T @ U.T
                        else:
                            # Materialize full weight for inference
                            W = (s_eff @ V).T @ U.T
                            h = x @ W
                        
                        add_out += h
                else:
                    # TRM recursion mode
                    # Project to singular value space
                    h = x @ V.T  # [b, s, r]
                    
                    # TRM recursion on last token
                    context = h[:, -1:, :]  # [b, 1, r]
                    b = context.shape[0]
                    
                    # Initialize or retrieve zH and zL in r_dim
                    zL = recursion_cache.get('zL', None)
                    if zL is None:
                        zL = self.svft_zL_init[adapter].unsqueeze(0).expand(b, -1).to(h.device)
                    zH = recursion_cache.get('zH', None)
                    if zH is None:
                        zH = self.svft_zH_init[adapter].unsqueeze(0).expand(b, -1).to(h.device)
                    
                    # TRM refines singular value deltas
                    zLs, zHs = self.trm(adapter, zL, zH, context)
                    
                    # Update cache for next layer
                    recursion_cache['zL'] = zLs[:, -1, :]  # [b, r]
                    recursion_cache['zH'] = zHs[:, -1, :]  # [b, r]
                    
                    # Apply refined singular values
                    sd_values = zHs * F.sigmoid(self.svft_gate[adapter])
                    s_eff = self.get_sparse_s_eff(adapter, sd_values[:, -1, :])
                    
                    # Complete transformation: h @ s_eff @ U^T
                    if self.training:
                        h = h @ s_eff.T @ U.T
                    else:
                        W = (s_eff @ V).T @ U.T
                        h = x @ W
                    
                    add_out += h

            result = base_out + add_out.to(base_out.dtype)

        result = result.to(previous_dtype)
        return result

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Merge not implemented for TRM SVFT yet")

    def unmerge(self) -> None:
        raise NotImplementedError("Unmerge not implemented for TRM SVFT yet")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmsvft." + rep


class TRMSvftLinear(nn.Module, TRMSvftLayer):
    """TRM SVFT implemented in a dense layer"""
    
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        svft_config: TRMSvftConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        TRMSvftLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, svft_config=svft_config, **kwargs)

    def forward(self, hidden_states: Float[Tensor, 'b s h'], *args: Any, **kwargs: Any) -> Float[Tensor, 'b s h']:
        """Forward pass - delegates to TRMSvftLayer.forward"""
        return TRMSvftLayer.forward(self, hidden_states, *args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmsvft." + rep


class TRMSvftModel:
    """
    TRM SVFT Model - handles adapter injection into base model.
    Follows PEFT patterns but simplified since SVFT isn't in PEFT core.
    """
    
    @staticmethod
    def _create_new_module(svft_config, adapter_name, target, **kwargs):
        """Create TRMSvftLinear for Linear layers."""
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = TRMSvftLinear(
                target, 
                adapter_name, 
                svft_config=svft_config,
                **kwargs
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported for TRM SVFT. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module
