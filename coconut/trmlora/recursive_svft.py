"""
TRM SVFT adapter - combines SVFT (Singular Value Fine-Tuning) with TRM recursion.

SVFT decomposes weights via SVD: W = U @ S @ V^T
- U, V are frozen singular vectors (orthonormal bases)
- S is diagonal singular values (frozen as s0)
- sd is sparse learnable delta to S (controlled by gate)
- TRM recursively refines sd in the singular value space

This is similar to TRM DeLoRA but uses sparse singular value updates instead of low-rank deltas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from jaxtyping import Float
from einops import repeat, rearrange
from peft.tuners.tuners_utils import BaseTunerLayer, BaseTuner
from peft.config import PeftConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType
from peft.utils.other import get_pattern_key

from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
)

from .bnb_utils import cast_adapter_input, cast_adapter_output
from .trm_adapter import L_net, trm_recursion


@dataclass
class TRMSvftAConfig(PeftConfig):
    """
    Configuration for TRM SVFT adapter.

    Config from https://github.com/VijayLingam95/SVFT/blob/8303115d45868712f952e6a847735bb59b1a9f18/MetaMath/run_math.sh#L29
    """
    # SVFT-specific parameters
    r: int = field(default=8, metadata={"help": "SVD rank (None = full rank)"})
    # NOTE: off_diag disabled - diagonal-only (Plain SVFT) for simplicity and parameter efficiency
    # Paper shows full-rank diagonal outperforms low-rank banded for same param count
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
        # if self.target_modules is None:
        #     self.target_modules = ["q_proj", "v_proj"]


def dense_sparse_mm(dense, sparse_T):
    """
    Compute dense @ sparse where:
    - dense: [b, s, r] or [b*s, r]
    - sparse_T: [r, r] (already transposed)
    
    Returns: [b, s, r] or [b*s, r]
    """
    shape = dense.shape
    dense_2d = dense.reshape(-1, dense.shape[-1])  # [b*s, r]
    
    # Want: dense_2d @ sparse_T
    # Compute: (sparse_T @ dense_2d.T).T
    result = torch.sparse.mm(sparse_T, dense_2d.t()).t()
    
    return result.reshape(shape)

class TRMSvftLayer(BaseTunerLayer):
    """
    TRM SVFT layer that wraps a base layer and applies TRM-enhanced SVFT.
    
    SVFT decomposes W = U @ S @ V^T where:
    - U, V are frozen orthonormal bases from SVD
    - S = s0 + sd where s0 is frozen diagonal, sd is diagonal learnable delta
    - TRM recursively refines sd in r-dimensional singular value space
    
    NOTE: Currently diagonal-only (Plain SVFT). Off-diagonal variants disabled for simplicity.
    Paper shows full-rank diagonal outperforms low-rank banded at same parameter count.

    Code from https://github.com/VijayLingam95/SVFT/blob/8303115d45868712f952e6a847735bb59b1a9f18/svft/svft_layers.py
    """
    
    adapter_layer_names = ("svft_sd", "svft_gate", "svft_l_nets")
    other_param_names = ("svft_u", "svft_v", "svft_s0", "svft_configs", "svft_zL_init", "svft_zH_init", "svft_s0_row", "svft_s0_col", "svft_sd_row", "svft_sd_col")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # BaseTunerLayer.__init__(self, base_layer)
        
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
        self.svft_configs: Dict[str, TRMSvftAConfig] = {}
        self.svft_l_nets = nn.ModuleDict({})
        
        # Mark the weight as unmerged
        self._disable_adapters = False

        # Marker for Coconut to find TRM layers
        self._recursion_cache = None
        
        self._active_adapter = None

    def update_layer(
        self,
        adapter_name: str,
        svft_config: TRMSvftAConfig,
        **kwargs
    ) -> None:
        """
        Initialize SVFT adapter on this layer.
        """
        if adapter_name in self.svft_u:
            return  # Already initialized
        
        self.svft_configs[adapter_name] = svft_config
        
        # Compute SVD of base weight
        base_weight = self.get_base_layer().weight  # [out, in]
        device = base_weight.device
        
        U, S, Vh = torch.linalg.svd(base_weight.float(), full_matrices=False)  # U: [out, min], S: [min], Vh: [min, in]
        # base_weight.half().cpu()  # free up GPU memory
        
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
        # s0_indices = torch.sparse.spdiags(S, torch.LongTensor([0]), (r, r)).coalesce().indices()
        s0_indices = torch.stack([torch.arange(r), torch.arange(r)])
        self.svft_s0_row[adapter_name] = s0_indices[0]
        self.svft_s0_col[adapter_name] = s0_indices[1]
        
        # Create diagonal pattern for sd (Plain SVFT)
        # k = r for diagonal-only, simplifies TRM to operate in r-dimensional space
        k = r
        sd_indices = torch.stack([torch.arange(r, device=device), torch.arange(r, device=device)])
        self.svft_sd_row[adapter_name] = sd_indices[0]
        self.svft_sd_col[adapter_name] = sd_indices[1]
        
        # Initialize learnable parameters
        sd = torch.zeros(k, device=device)
        nn.init.kaiming_normal_(sd[None, :])
        self.svft_sd[adapter_name] = nn.Parameter(sd, requires_grad=True)
        self.svft_gate[adapter_name] = nn.Parameter(torch.tensor([0.0], device=device), requires_grad=True)
        
        # Initialize TRM components
        self.svft_l_nets[adapter_name] = L_net(
            k,
            svft_config.l_layers,
            svft_config.num_heads,
            svft_config.expansion,
        )
        
        # Initialize TRM recursion states in r_dim
        zH = torch.empty(k, device=device)
        zL = torch.empty(k, device=device)
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

    
    def get_sparse_s_eff(self, adapter_name: str, sd_values = None) -> torch.sparse_coo_tensor:
        """
        Compute effective singular value matrix: s_eff = s0 + sd
        Returns PyTorch sparse COO tensor.
        """
        if sd_values is None:
            sd_values = self.svft_sd[adapter_name] * F.sigmoid(self.svft_gate[adapter_name])
        
        r = self.svft_s0[adapter_name].shape[0]
        
        # Create sparse tensors using native PyTorch
        sd_indices = torch.stack([self.svft_sd_row[adapter_name], self.svft_sd_col[adapter_name]])
        sd = torch.sparse_coo_tensor(sd_indices, sd_values.float(), (r, r))
        
        s0_indices = torch.stack([self.svft_s0_row[adapter_name], self.svft_s0_col[adapter_name]])
        s0 = torch.sparse_coo_tensor(s0_indices, self.svft_s0[adapter_name].float(), (r, r))
        

        # TODO I'd like a multiplicative option, like IA3, ROAD, or VERA
        s_eff = (s0 + sd).coalesce()
        
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
            
            # Store expected dtype for quantized models
            expected_dtype = base_out.dtype

            for adapter in self.active_adapters:
                if adapter not in self.svft_u:
                    continue

                # Cast input for quantized models
                x_cast = cast_adapter_input(x, self.svft_v[adapter])

                U = self.svft_u[adapter]
                V = self.svft_v[adapter]
                
                # Check if we're in steering mode (post-latent)
                steering_mode = recursion_cache.get('steering_mode', False)

                # Project to singular value space
                h_s = x_cast @ V.T  # [b, s, r]
                
                if steering_mode:
                    # Don't run TRM, just apply cached refined sd
                    zH = recursion_cache.get('zH')
                    zL = recursion_cache.get('zL')
                    zHs = zH.unsqueeze(1)  # [b, 1, r]

                    # TRM refines direction (operates on normalized space)
                    zLs, zHs = self.trm(adapter, zL, zH, h_s, h_cycles=1)  # zH is refined 1 time
                else:
                    # TRM recursion mode
                    
                    # For diagonal SVFT, k=r, so context is full singular value projection
                    b = h_s.shape[0]

                    # Initialize or retrieve zH and zL in r_dim (singular value space)
                    zL = recursion_cache.get('zL', None)
                    if zL is None:
                        zL = self.svft_zL_init[adapter].unsqueeze(0).expand(b, -1).to(x.device)
                    zH = recursion_cache.get('zH', None)
                    if zH is None:
                        zH = self.svft_zH_init[adapter].unsqueeze(0).expand(b, -1).to(x.device)
                    
                    # TRM refines all r singular value deltas
                    zLs, zHs = self.trm(adapter, zL, zH, h_s)
                    
                    # Update cache for next layer
                    recursion_cache['zL'] = zLs[:, -1, :]  # [b, r]
                    recursion_cache['zH'] = zHs[:, -1, :]  # [b, r]
                    
                # Apply refined singular values (diagonal matrix - no sparse ops needed)
                sd_values = zHs[:, -1, :] * F.sigmoid(self.svft_gate[adapter])  # [b, r]
                s_eff_diag = self.svft_s0[adapter] + sd_values  # [b, r] broadcast with [r]
                
                # Complete transformation: x @ V.T @ diag(s_eff) @ U.T
                # h_s is already x @ V.T: [b, s, r]
                # For diagonal matrix: h_s @ diag(s_eff) = h_s * s_eff (elementwise)
                h = (h_s * s_eff_diag.unsqueeze(1)) @ U.T  # [b, s, r] * [b, 1, r] -> [b, s, out]
                
                # Cast output for quantized models
                h = cast_adapter_output(h, expected_dtype)
                
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
        svft_config: TRMSvftAConfig,
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


class TRMSvftModel(BaseTuner):
    """
    TRM SVFT Model - handles adapter injection into base model.
    Inherits from BaseTuner to integrate with PEFT infrastructure.
    """
    prefix: str = "svft_"
    tuner_layer_cls = TRMSvftLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


    def _create_and_replace(
        self,
        svft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key 
        kwargs = {
            # "svft_config": svft_config,
        }

        if isinstance(target, TRMSvftLinear):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(svft_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
    
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
