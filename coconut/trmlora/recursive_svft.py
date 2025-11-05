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
from torch import Tensor
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from jaxtyping import Float
from peft.tuners.tuners_utils import BaseTunerLayer, BaseTuner
from peft.config import PeftConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
import bitsandbytes as bnb
from bitsandbytes.nn import Params4bit, Int8Params

from .trm_adapter import L_net, trm_seq


@dataclass
class TRMSvftAConfig(PeftConfig):
    """
    Configuration for TRM SVFT adapter.

    Config from https://github.com/VijayLingam95/SVFT/blob/8303115d45868712f952e6a847735bb59b1a9f18/MetaMath/run_math.sh#L29
    
    Hybrid SVD merging: Approximate full SVD cheaply. Principal (top-principal_rank SVD) captures base variance. Tail (low-rank random ortho basis to principal V, zero S init) merges tail info without full compute. Hypothesis: Principal leverages pretrain; tail recovers subtle patterns > pure top-k or random LoRA. Concat bases; single TRM on r=principal+tail (principal strong init, tail explores).
    """
    # SVFT-specific parameters
    r: int = field(default=32, metadata={"help": "Top-r SVD rank (principal directions)"})
    rotate_u: bool = field(
        default=False,
        metadata={"help": "Enable U rotation via TRM-learned Householder reflections"}
    )
    rotate_v: bool = field(
        default=True,
        metadata={"help": "Enable V rotation via TRM-learned Householder reflections"}
    )
    svft_mode: Literal["add", "mult"] = field(
        default="mult",
        metadata={
            "help": "SVFT mode: 'add' (S + exp(log_sd)), 'mult' (S * exp(log_sd))"
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
    
    adapter_layer_names = ("svft_l_nets", "svft_zL_init", "svft_zH_init", "svft_output_head_s", "svft_output_head_rot_v", "svft_output_head_rot_u")
    other_param_names = ("svft_u", "svft_v", "svft_s", "svft_w_res", "svft_configs", "svft_actual_r")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # BaseTunerLayer.__init__(self, base_layer)
        
        # SVFT components (per adapter) - match reference naming
        self.svft_u = BufferDict({})  # U: [d_out, r] - frozen
        self.svft_v = BufferDict({})  # V: [d_in, r] - frozen
        self.svft_s = BufferDict({})  # S: [r] - frozen
        self.svft_w_res = BufferDict({})  # W_res: [d_out, d_in] - frozen residual
        
        # Per-adapter output heads (modules) for mixing zH
        self.svft_output_head_s = nn.ModuleDict({})  # zH → S scaling
        self.svft_output_head_rot_v = nn.ModuleDict({})  # zH → V rotation params
        self.svft_output_head_rot_u = nn.ModuleDict({})  # zH → U rotation params

        # TRM components (single for combined r)
        self.svft_zL_init = nn.ParameterDict({})
        self.svft_zH_init = nn.ParameterDict({})
        self.svft_configs: Dict[str, TRMSvftAConfig] = {}
        self.svft_actual_r: Dict[str, int] = {}  # Store actual r used (may be < config.r for small matrices)
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
        Initialize SVFT adapter with simple top-r SVD + residual (like reference).
        No tail merging - residual captures what's left after top-r.
        """
        if adapter_name in self.svft_u:
            return  # Already initialized
        
        self.svft_configs[adapter_name] = svft_config
        
        # Get base weight
        base_weight = self.get_base_layer().weight
        
        # Dequantize if needed for full-precision SVD
        if isinstance(base_weight, Params4bit):
            base_weight = bnb.functional.dequantize_4bit(base_weight.data, base_weight.quant_state)
        elif isinstance(base_weight, Int8Params):
            base_weight = bnb.functional.dequantize_8bit(base_weight.data, base_weight.quant_state)
        
        base_weight = base_weight.float()  # [out, in]
        device = base_weight.device
        
        r_config = svft_config.r
        
        # Simple top-r SVD (like reference)
        U_full, S_full, Vh_full = torch.linalg.svd(base_weight, full_matrices=False)
        
        # Use min(r_config, actual_rank) to handle small matrices
        r = min(r_config, U_full.shape[1])
        
        U = U_full[:, :r]  # [d_out, r]
        S = S_full[:r]     # [r]
        Vh = Vh_full[:r, :]  # [r, d_in]
        V = Vh.T           # [d_in, r]
        
        # Compute residual (reference style)
        W_principal = U @ torch.diag(S) @ Vh
        W_res = base_weight - W_principal
        
        # Store frozen components
        self.svft_u[adapter_name] = U.clone().detach().contiguous()
        self.svft_v[adapter_name] = V.clone().detach().contiguous()
        self.svft_s[adapter_name] = S.clone().detach().contiguous()
        self.svft_w_res[adapter_name] = W_res.clone().detach().contiguous()
        self.svft_actual_r[adapter_name] = r  # Store actual r used
        
        # Initialize TRM components
        # Split zH into 3r dimensions: [S_region, U_region, V_region]
        # Each region is independently refined by TRM
        k = r# * (1 + int(svft_config.rotate_u) + int(svft_config.rotate_v))  # Total dimension
        self.svft_l_nets[adapter_name] = L_net(
            k,
            svft_config.l_layers,
            svft_config.num_heads,
            svft_config.expansion,
        )
        
        # Initialize TRM recursion states
        zH = torch.empty(k, device=device)
        zL = torch.empty(k, device=device)
        torch.nn.init.trunc_normal_(zH, std=1.0)
        torch.nn.init.trunc_normal_(zL, std=1.0)
        self.svft_zL_init[adapter_name] = nn.Parameter(zL, requires_grad=True)
        self.svft_zH_init[adapter_name] = nn.Parameter(zH, requires_grad=True)

        # Why do we need output heads? zH is recursively refined and needs stability, but the interventions on S/U/V need dynamic range. Output heads allow the model to translate between these spaces instead of struggling to do both.
        # Initialize output heads

        self.svft_output_head_s[adapter_name] = nn.Linear(k, k, bias=True)
        nn.init.trunc_normal_(self.svft_output_head_s[adapter_name].weight, std=0.02)
        
        if svft_config.rotate_v:
            # Output r Householder vectors for V rotation
            self.svft_output_head_rot_v[adapter_name] = nn.Linear(k, k, bias=False)
            nn.init.trunc_normal_(self.svft_output_head_rot_v[adapter_name].weight, std=0.02)
        
        if svft_config.rotate_u:
            # Output r Householder vectors for U rotation
            self.svft_output_head_rot_u[adapter_name] = nn.Linear(k, k, bias=False)
            nn.init.trunc_normal_(self.svft_output_head_rot_u[adapter_name].weight, std=0.02)

    def trm(self, adapter_name: str, zL, zH, context_hs, latent_mask, h_cycles=None):
        """Wrapper around trm_recursion with adapter-specific config."""
        svft_config = self.svft_configs[adapter_name]
        if h_cycles is None:
            h_cycles = svft_config.h_cycles
        
        return trm_seq(
            l_net=self.svft_l_nets[adapter_name],
            zL=zL,
            zH=zH,
            x=context_hs,
            latent_mask=latent_mask,
            l_cycles=svft_config.l_cycles,
            h_cycles=h_cycles,
        )

    def apply_householder_rotation(
        self,
        basis: Float[Tensor, "d r"],
        rot_params: Float[Tensor, "b s r"],
    ) -> Float[Tensor, "b s d r"]:
        """
        Rotate basis using Householder reflections (HRA-style).
        
        Given r Householder vectors, builds rotation R = H_r @ ... @ H_1
        where H_i = I - 2*u_i*u_i^T, then computes basis_rot = basis @ R.
        
        Args:
            basis: Frozen basis matrix (U or V) [d, r]
            rot_params: Normalized Householder vectors from TRM [b, s, r]
            
        Returns:
            basis_rot: Rotated basis [b, s, d, r]
        """
        b, s, r = rot_params.shape
        device = basis.device
        dtype = basis.dtype
        
        # Start with identity rotation for each (b, s) position
        R = torch.eye(r, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(b, s, r, r).clone()
        
        # Apply r Householder reflections sequentially: R_new = R_old @ H_i
        # where H_i = I - 2*u_i @ u_i^T
        for i in range(r):
            ui = rot_params[:, :, i].unsqueeze(-1)  # [b, s, 1]
            # Compute R @ u_i to get i-th column transform
            R_ui = R[:, :, :, i:i+1]  # [b, s, r, 1] - i-th column of current R
            # H_i @ R = (I - 2*u_i*u_i^T) @ R = R - 2*u_i*(u_i^T @ R)
            # Since u_i is a column vector, u_i^T @ R means scaling each column of R by u_i[j]
            # Actually: R @ H_i = R @ (I - 2*u_i*u_i^T) = R - 2*(R@u_i)@u_i^T
            R = R - 2 * R_ui * ui.unsqueeze(-2)  # [b, s, r, r]
        
        # Apply rotation to basis: basis_rot = basis @ R
        basis_expanded = basis.unsqueeze(0).unsqueeze(0)  # [1, 1, d, r]
        basis_rot = torch.matmul(basis_expanded, R)  # [b, s, d, r]
        
        return basis_rot

    def get_delta(self, x, adapter: str) -> torch.Tensor:
        """
        Compute adapter delta with TRM-learned S scaling and V rotation.
        
        V rotation via Householder reflections (if enabled)
        S scaling via TRM output (as before)
        U can have learnable delta (weight decay pulls toward frozen init)
        """
        b = x.shape[0]
        svft_config = self.svft_configs[adapter]

        if self._recursion_cache is None:
            recursion_cache = {}
        else:
            recursion_cache = self._recursion_cache

        # Initialize or retrieve zH and zL in r_dim
        zL = recursion_cache.get('zL', None)
        if zL is None:
            zL = self.svft_zL_init[adapter].unsqueeze(0).expand(b, -1).to(x.device)
        zH = recursion_cache.get('zH', None)
        if zH is None:
            zH = self.svft_zH_init[adapter].unsqueeze(0).expand(b, -1).to(x.device)

        # Get frozen SVD components
        U = self.svft_u[adapter]  # [d_out, r]
        V = self.svft_v[adapter]  # [d_in, r]
        S = self.svft_s[adapter]  # [r]
        W_res = self.svft_w_res[adapter]  # [d_out, d_in]
        
        # Project to singular value space (using frozen V initially)
        x_v = x @ V  # [b, s, d_in] @ [d_in, r] -> [b, s, r]

        latent_mask = recursion_cache['latent_mask']
        
        zLs, zHs = self.trm(adapter, zL, zH, x_v, latent_mask=latent_mask)

        # Save last one for next step
        recursion_cache['zL'] = zLs[:, -1, :]  # [b, r]
        recursion_cache['zH'] = zHs[:, -1, :]  # [b, r]
        
        # Get output heads from zH
        zH_s = self.svft_output_head_s[adapter](zHs)  # [b, s, r] - S scaling params
        
        # Rotate V basis if enabled (HRA-style with r Householder vectors)
        if svft_config.rotate_v:
            zH_rot_v = self.svft_output_head_rot_v[adapter](zHs)  # [b, s, r]
            # Normalize Householder vectors
            zH_rot_v = zH_rot_v / (zH_rot_v.norm(dim=-1, keepdim=True) + 1e-8)
            V_rot = self.apply_householder_rotation(V, zH_rot_v)  # [b, s, d_in, r]
            # Reproject with rotated basis
            x_v = torch.einsum('bsd,bsdr->bsr', x, V_rot)  # [b, s, d_in] @ [b, s, d_in, r] -> [b, s, r]
        
        # Apply S scaling (zH_s as log_sd for positive scaling)
        S_expanded = S.unsqueeze(0).unsqueeze(0)  # [1, 1, r]
        mode = svft_config.svft_mode
        
        log_sd = zH_s  # Interpret TRM output as log-space for positivity
        if mode == "mult":
            # Multiplicative: S_eff = S * exp(log_sd)
            s_eff = S_expanded * torch.exp(log_sd)
        elif mode == "add":
            # Additive: S_eff = S + exp(log_sd)
            s_eff = S_expanded + torch.exp(log_sd)
        else:
            raise ValueError(f"Unknown svft_mode: {mode}")
        
        # Soft clamp for stability (prevent explosion)
        def soft_clamp(x, n=10.0):
            return n * torch.tanh(x / n)
        max_magnitude = 10.0 * S_expanded + 1
        s_eff = soft_clamp(s_eff, n=max_magnitude)
  
        # Apply S scaling
        x_v_scaled = x_v * s_eff  # [b, s, r]
        
        # Rotate U basis if enabled (output space, HRA-style)
        if svft_config.rotate_u:
            zH_rot_u = self.svft_output_head_rot_u[adapter](zHs)  # [b, s, r]
            # Normalize Householder vectors
            zH_rot_u = zH_rot_u / (zH_rot_u.norm(dim=-1, keepdim=True) + 1e-8)
            U_rot = self.apply_householder_rotation(U, zH_rot_u)  # [b, s, d_out, r]
            # Project with rotated basis: x_v_scaled @ U_rot^T
            h_svd = torch.einsum('bsr,bsor->bso', x_v_scaled, U_rot)  # [b, s, r] @ [b, s, d_out, r]^T -> [b, s, d_out]
        else:
            # Project with frozen U
            h_svd = x_v_scaled @ U.T  # [b, s, r] @ [r, d_out] -> [b, s, d_out]
        
        h_res = x @ W_res.T  # [b, s, d_in] @ [d_in, d_out] -> [b, s, d_out]
        
        return h_svd + h_res

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

            # SVFT replaces base layer output with modified SVD reconstruction
            # Output: x @ V @ diag(S_eff) @ U.T + x @ W_res.T
            result = None
            for adapter in self.active_adapters:
                if adapter not in self.svft_u:
                    continue

                h = self.get_delta(x, adapter)
                
                if result is None:
                    result = h
                else:
                    result += h  # Multiple adapters (unlikely)
            
            if result is None:
                result = self.base_layer(x, *args, **kwargs)

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
