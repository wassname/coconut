"""
TRM SVFT adapter - combines SVFT (Singular Value Fine-Tuning) with TRM recursion.

SVFT decomposes weights via SVD: W = U @ S @ V^T
- U, V are frozen singular vectors (orthonormal bases)
- S is diagonal singular values (frozen as s0)
- sd is sparse learnable delta to S (controlled by gate)
- TRM recursively refines sd in r-dimensional singular value space

This is similar to TRM DeLoRA but uses sparse singular value updates instead of low-rank deltas.

Hybrid SVD merging: Approximate full SVD cheaply. Principal (top-principal_rank SVD) captures base variance. Tail (low-rank random ortho basis to principal V, zero S init) merges tail info without full compute. Hypothesis: Principal leverages pretrain; tail recovers subtle patterns > pure top-k or random LoRA. Concat bases; single TRM on r=principal+tail (principal strong init, tail explores).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from jaxtyping import Float
from einops import repeat, rearrange
from peft.tuners.tuners_utils import BaseTunerLayer, BaseTuner
from peft.config import PeftConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType
from peft.utils.other import get_pattern_key
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit, Int8Params

from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
)

from .trm_adapter import L_net, trm_recursion


@dataclass
class TRMSvftAConfig(PeftConfig):
    """
    Configuration for TRM SVFT adapter.

    Config from https://github.com/VijayLingam95/SVFT/blob/8303115d45868712f952e6a847735bb59b1a9f18/MetaMath/run_math.sh#L29
    
    Hybrid SVD merging: Approximate full SVD cheaply. Principal (top-principal_rank SVD) captures base variance. Tail (low-rank random ortho basis to principal V, zero S init) merges tail info without full compute. Hypothesis: Principal leverages pretrain; tail recovers subtle patterns > pure top-k or random LoRA. Concat bases; single TRM on r=principal+tail (principal strong init, tail explores).
    """
    # SVFT-specific parameters
    r: int = field(default=19, metadata={"help": "Rank, includes Top-k SVD rank for principal directions (base variance), and tail_rank for low-rank approx of remaining vectors (subtle info)"})
    tail_rank: int = field(default=4, metadata={"help": "Low-rank approx rank for tail merging (ortho random basis; subtle info)"})
    # NOTE: off_diag disabled - diagonal-only (Plain SVFT) for simplicity and parameter efficiency
    # Paper shows full-rank diagonal outperforms low-rank banded for same param count
    fill_orthonormal: bool = field(
        default=False, 
        metadata={"help": "Fill beyond r with random orthonormal (disabled; tail replaces it)"}
    )
    learnable_u: bool = field(
        default=True,
        metadata={"help": "Make U learnable via delta parameterization (U_eff = U_init + U_delta). Weight decay pulls U_delta→0."}
    )
    svft_mode: Literal["replace_add", "replace_mul", "adapter_add", "adapter_mult"] = field(
        default="adapter_add",
        metadata={
            "help": "SVFT mode: replace_add (s0+sd, replace base), replace_mul (s0*(1+sd), replace base), adapter_add (sd only, add to base), adapter_mult (sd*s0 only, add to base)"
        }
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
        assert self.r > self.tail_rank, "Total rank r must be greater than tail_rank"
        self.principal_rank = self.r - self.tail_rank
        # self.r = self.principal_rank + self.tail_rank  # Total rank
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


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
    
    adapter_layer_names = ("svft_l_nets", "svft_zL_init", "svft_zH_init", "svft_u_delta", "svft_output_head")
    other_param_names = ("svft_u_init", "svft_v", "svft_s0", "svft_configs")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # BaseTunerLayer.__init__(self, base_layer)
        
        # SVFT components (per adapter)
        self.svft_u_init = BufferDict({})  # Frozen SVD U (init)
        self.svft_u_delta = nn.ParameterDict({})  # Learnable delta: U_eff = U_init + U_delta
        self.svft_v = nn.ParameterDict({})
        self.svft_s0 = BufferDict({})
        # Per-adapter output heads (modules) for mixing zH
        self.svft_output_head = nn.ModuleDict({})

        # TRM components (single for combined r)
        self.svft_zL_init = nn.ParameterDict({})
        self.svft_zH_init = nn.ParameterDict({})
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
        Initialize SVFT adapter on this layer with hybrid SVD merging (concat principal + tail bases).
        """
        if adapter_name in self.svft_u_init:
            return  # Already initialized
        
        self.svft_configs[adapter_name] = svft_config
        
        # Compute SVD of base weight
        base_weight = self.get_base_layer().weight
        
        # Dequantize if needed for full-precision SVD
        if isinstance(base_weight, Params4bit):
            base_weight = bnb.functional.dequantize_4bit(base_weight.data, base_weight.quant_state)
        elif isinstance(base_weight, Int8Params):
            base_weight = bnb.functional.dequantize_8bit(base_weight.data, base_weight.quant_state)
        
        base_weight = base_weight.float()  # [out, in]
        device = base_weight.device
        
        principal_r = svft_config.principal_rank
        tail_r = svft_config.tail_rank
        r = principal_r + tail_r
        
        # Principal: top-principal_rank SVD
        U_p, S_p, Vh_p = torch.linalg.svd(base_weight, full_matrices=False)
        U_p = U_p[:, :principal_r]
        Vh_p = Vh_p[:principal_r, :]
        S_p = S_p[:principal_r]
        
        # Tail approx: random low-rank basis ortho to principal V_p
        if tail_r > 0:
            # Random init for tail V basis [tail_r, in]
            random_v_tail = torch.randn(tail_r, base_weight.shape[1], device=device)
            # Orthogonalize to principal V_p span (project out principal component)
            inner_proj = Vh_p @ random_v_tail.T  # [principal_r, in] @ [in, tail_r] -> [principal_r, tail_r]
            proj_principal = Vh_p.T @ inner_proj  # [in, principal_r] @ [principal_r, tail_r] -> [in, tail_r]
            ortho_v_tail = random_v_tail.T - proj_principal  # Subtract projection [in, tail_r]
            Q_v_tail, _ = torch.linalg.qr(ortho_v_tail)  # Ortho basis [in, tail_r]
            Vh_tail = Q_v_tail.T  # [tail_r, in]
            
            # Random ortho U_tail [out, tail_r]
            U_tail = torch.randn(base_weight.shape[0], tail_r, device=device)
            nn.init.orthogonal_(U_tail)
            S_tail = torch.zeros(tail_r, device=device)  # Zero init for tail

            # init S_tail to small values?
            nn.init.uniform_(S_tail, a=1e-5, b=1e-4)
        else:
            Vh_tail = None
            U_tail = None
            S_tail = None
        
        # Concat principal + tail for combined basis
        U = torch.cat([U_p, U_tail], dim=1) if tail_r > 0 else U_p
        Vh = torch.cat([Vh_p, Vh_tail], dim=0) if tail_r > 0 else Vh_p
        S = torch.cat([S_p, S_tail]) if tail_r > 0 else S_p
        
        # Optionally fill remaining with orthonormal (if r < full)
        full_min = min(base_weight.shape)
        if svft_config.fill_orthonormal and r < full_min:
            diff_rank = full_min - r
            U_fill = torch.randn(base_weight.shape[0], diff_rank, device=device)
            nn.init.orthogonal_(U_fill)
            Vh_fill = torch.randn(diff_rank, base_weight.shape[1], device=device)
            nn.init.orthogonal_(Vh_fill)
            U = torch.cat([U, U_fill], dim=1)
            Vh = torch.cat([Vh, Vh_fill], dim=0)
            S = torch.cat([S, torch.zeros(diff_rank, device=device)])
            r = S.shape[0]
        
        # Store combined U, Vh, S
        self.svft_u_init[adapter_name] = U.clone().detach().contiguous()  # Frozen
        self.svft_u_delta[adapter_name] = nn.Parameter(
            torch.zeros_like(U), 
            requires_grad=svft_config.learnable_u
        )
        self.svft_v[adapter_name] = nn.Parameter(Vh.clone().detach().contiguous(), requires_grad=False)
        self.svft_s0[adapter_name] = S.clone().detach().contiguous()
        
        # Initialize learnable parameters based on mode
        mode = svft_config.svft_mode
                
        # Initialize TRM components: single L_net on combined r
        k = r
        self.svft_l_nets[adapter_name] = L_net(
            k,
            svft_config.l_layers,
            svft_config.num_heads,
            svft_config.expansion,
        )
        
        # Initialize TRM recursion states in total r_dim
        zH = torch.empty(k, device=device)
        zL = torch.empty(k, device=device)
        torch.nn.init.trunc_normal_(zH, std=1.0)
        torch.nn.init.trunc_normal_(zL, std=1.0)
        self.svft_zL_init[adapter_name] = nn.Parameter(zL, requires_grad=True)
        self.svft_zH_init[adapter_name] = nn.Parameter(zH, requires_grad=True)

        # Initialize output head for zH
        self.svft_output_head[adapter_name] = nn.Linear(k, k, bias=False)
        nn.init.trunc_normal_(self.svft_output_head[adapter_name].weight, std=0.02)

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

    def get_delta(self, x, adapter: str) -> torch.Tensor:
        """
        Compute adapter delta with ΔU parameterization.
        U_effective = U_init + U_delta (weight decay on U_delta naturally pulls U → U_init)
        """

        if self._recursion_cache is None:
            recursion_cache = {}
        else:
            recursion_cache = self._recursion_cache

        # Effective U = U_init (frozen SVD) + U_delta (learnable adaptation)
        U = self.svft_u_init[adapter] + self.svft_u_delta[adapter]
        V = self.svft_v[adapter]
        
        # DeLoRA pattern: Normalize INPUT by down-projection matrix norm
        # Down-project to singular value space
        x_v = x @ V.T  # [b, s, r]
        
        # Normalize by V's magnitude (like DeLoRA's A norm)
        # Vn = torch.clamp(V.norm(dim=1), min=1e-6)  # [r] - norm of each row of V
        # x_v_normalized = x_v / Vn.unsqueeze(0).unsqueeze(0)  # [b, s, r] - unit norm per component
        
        # Check if we're in steering mode (post-latent)
        steering_mode = recursion_cache.get('steering_mode', False)
        
        if steering_mode:
            # Don't run TRM, just apply cached refined sd
            zH = recursion_cache.get('zH')
            zL = recursion_cache.get('zL')
            zLs, zHs = self.trm(adapter, zL, zH, x_v, h_cycles=1)
            zHs = self.svft_output_head[adapter](zHs)  # Apply head in steering
        else:
            # TRM recursion mode
            b = x_v.shape[0]

            # Initialize or retrieve zH and zL in r_dim
            zL = recursion_cache.get('zL', None)
            if zL is None:
                zL = self.svft_zL_init[adapter].unsqueeze(0).expand(b, -1).to(x.device)
            zH = recursion_cache.get('zH', None)
            if zH is None:
                zH = self.svft_zH_init[adapter].unsqueeze(0).expand(b, -1).to(x.device)
            
            # TRM refines in normalized space (like DeLoRA)
            zLs, zHs = self.trm(adapter, zL, zH, x_v[:, -1:])
            zHs = self.svft_output_head[adapter](zHs)  # Apply head after TRM
            
            # Update cache for next layer
            recursion_cache['zL'] = zLs[:, -1, :]  # [b, r]
            recursion_cache['zH'] = zHs[:, -1, :]  # [b, r]
            
        # Don't normalize zHs - it contains learned per-component magnitudes!
        S0 = self.svft_s0[adapter]  # [r] - base singular values
        
        # Apply mode-specific transformation
        mode = self.svft_configs[adapter].svft_mode
        S0_expanded = S0.unsqueeze(0).unsqueeze(0)  # [1, 1, r]
        
        if mode == "adapter_add":
            # Additive delta: exp for per-component expressiveness
            # Scaling by tanh ensure delta [-s0, s0] range, which means we can't flip sign of singular values
            s_eff = torch.tanh(zHs) * S0_expanded 
            
        elif mode == "adapter_mult":
            # Multiplicative delta: softplus for smooth positive scaling
            sd = F.softplus(zHs)
            s_eff = sd * S0_expanded
            
        elif mode == "replace_add":
            # Replace via addition: softplus for positive values
            sd = F.softplus(zHs)
            s_eff = S0_expanded + sd
            
        elif mode == "replace_mul":
            # Replace via multiplication: softplus for positive scaling factor
            sd = F.softplus(zHs)
            s_eff = S0_expanded * (1.0 + sd)
            
        else:
            raise ValueError(f"Unknown svft_mode: {mode}")
        
        # Final safety clamp
        if mode.startswith("replace_"):
            s_eff_diag = torch.clamp(s_eff, min=1e-6)  # Ensure positive for valid SVD
        else:
            # Adapter modes: bound magnitude to prevent instability
            max_magnitude = 10.0 * S0_expanded
            s_eff_diag = torch.clamp(s_eff, min=-max_magnitude, max=max_magnitude)
        
        # Complete transformation: x @ V.T @ diag(s_eff) @ U.T
        # x_v is already x @ V.T: [b, s, r]
        # For diagonal matrix: x_v @ diag(s_eff) = x_v * s_eff (elementwise)
        h = (x_v * s_eff_diag) @ U.T  # [b, s, r] * [b, s, r] -> [b, s, out]

        return h

    def forward(self, x: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
        previous_dtype = x.dtype
        
        # Use injected cache from Coconut.recursion_context() if available
        assert len(self.active_adapters) <= 1, "TRM SVFT currently supports only one active adapter at a time."

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(x, *args, **kwargs).to(previous_dtype)

            # Check mode from first active adapter
            adapter = self.active_adapters[0]
            mode = self.svft_configs[adapter].svft_mode if adapter in self.svft_configs else "replace_add"
            
            if mode.startswith("replace_"):
                # Replacement mode - compute x @ (U @ S_eff @ V.T).T directly
                # This replaces the base layer output entirely (like original SVFT)
                result = None
                for adapter in self.active_adapters:
                    if adapter not in self.svft_u_init:
                        continue

                    h = self.get_delta(x, adapter)
                    
                    if result is None:
                        result = h
                    else:
                        result += h  # Multiple oadapters (unlikely)
                
                if result is None:
                    result = self.base_layer(x, *args, **kwargs)
            else:
                # Adapter mode - add delta to base layer output
                base_out = self.base_layer(x, *args, **kwargs)
                add_out = torch.zeros_like(base_out)

                for adapter in self.active_adapters:
                    if adapter not in self.svft_u_init:
                        continue

                    h = self.get_delta(x, adapter)
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

    def forward(self, hidden_states: Float[Tensor, '...'], *args: Any, **kwargs: Any) -> Float[Tensor, '...']:
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
