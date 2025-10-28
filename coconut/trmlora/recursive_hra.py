"""
This subclasses hra, adapted from recursive_delora.py to use HRA instead of DeLoRA, integrating TRM recursion using HRA's u as low-rank projection basis for refinement delta added to HRA output.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from jaxtyping import Float

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.hra.layer import HRALayer
from peft.tuners.hra.model import HRAModel
from peft.tuners.hra.config import HRAConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType

from .trm_adapter import L_net, trm_recursion

@dataclass
class TRMHraAConfig(HRAConfig):
    """
    Configuration for TRM HRA adapter.
    Inherits from HRAConfig to get all standard HRA features.
    """
    # HRA-specific scaling for TRM delta (since HRA is orthogonal, alpha scales the additive refinement)
    hra_alpha: int = field(default=16, metadata={"help": "Scaling factor for TRM refinement delta"})

    # TRM-specific parameters
    l_cycles: int = field(default=6, metadata={"help": "Number of L_net cycles per H cycle"})
    h_cycles: int = field(default=2, metadata={"help": "Number of H cycles"})
    expansion: float = field(default=2.67, metadata={"help": "TRM expansion factor"})
    l_layers: int = field(default=2, metadata={"help": "Number of L_net layers"})
    num_heads: int = field(default=8, metadata={"help": "Number of attention heads"})
    update_mode: str = field(default="hra", metadata={"help": "Update mode: 'hra'"})
    transcoder_layers: int = field(default=2, metadata={"help": "Number of transcoder layers"})
    cycles: int = field(default=1, metadata={"help": "Additional refinement cycles"})

    def __post_init__(self):
        super().__post_init__()
        # Ensure peft_type is HRA
        self.peft_type = 'TRMHRA'
        
        # Validate TRM-specific constraints
        if self.h_cycles < 1:
            raise ValueError(f"h_cycles must be >= 1, got {self.h_cycles}")
        if self.l_cycles < 1:
            raise ValueError(f"l_cycles must be >= 1, got {self.l_cycles}")

class TRMHraLayer(HRALayer):
    """
    TRM HRA layer that wraps a base layer and overrides forward for HRA + TRM refinement.
    Subclasses HRALayer for proper PEFT integration.
    """
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "hra_u",
        "l_nets",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "apply_GS",
        "zL_init",
        "zH_init",
        "hra_alpha",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer, **kwargs)
        # TRM-specific state, prefixed with hra for saving
        self.hra_zL_init = BufferDict({})
        self.hra_zH_init = BufferDict({})
        self.hra_configs: Dict[str, TRMHraAConfig] = {}
        self.hra_l_nets = nn.ModuleDict({})
        self.hra_alpha: Dict[str, float] = {}
        
        # Marker for Coconut to find TRM layers
        self._recursion_cache = None  # Injected by Coconut.recursion_context()
    
    def update_layer(
        self,
        adapter_name: str,
        hra_config: TRMHraAConfig,
        r: int,
        apply_GS: bool,
        init_weights: bool = True,
        inference_mode: bool = False,
        **kwargs
    ) -> None:
        """
        Extend HRALayer.update_layer to add TRM-specific components.
        Calls parent to handle standard HRA setup (u, etc).
        """
        # Call parent to handle standard HRA setup
        super().update_layer(
            adapter_name=adapter_name,
            r=r,
            apply_GS=apply_GS,
            init_weights=init_weights,
            inference_mode=inference_mode,
            **kwargs
        )
        
        # Add TRM-specific components
        self.hra_configs[adapter_name] = hra_config
        self.hra_alpha[adapter_name] = hra_config.hra_alpha

        device = self.get_base_layer().weight.device
        
        self.hra_l_nets[adapter_name] = L_net(
            r,
            hra_config.l_layers,
            hra_config.num_heads,
            hra_config.expansion,
        )

        # Initialize TRM recursion states in r_dim
        zH = torch.empty(r, device=device)
        zL = torch.empty(r, device=device)
        torch.nn.init.trunc_normal_(zH, std=1.0)
        torch.nn.init.trunc_normal_(zL, std=1.0)
        self.hra_zL_init[adapter_name] = zL
        self.hra_zH_init[adapter_name] = zH

    def trm(self, adapter_name: str, zL: Float[Tensor, 'b h'], zH: Float[Tensor, 'b h'], context_hs: Float[Tensor, 'b h'], h_cycles=None) -> tuple[Float[Tensor, 'b h'], Float[Tensor, 'b h']]:
        """Wrapper around trm_recursion with adapter-specific config."""
        hra_config = self.hra_configs[adapter_name]
        if h_cycles is None:
            h_cycles = hra_config.h_cycles
        
        # trm_recursion expects [b, s, r], so add sequence dimension
        context = context_hs.unsqueeze(1)  # [b, 1, r]
        
        zLs, zHs = trm_recursion(
            l_net=self.hra_l_nets[adapter_name],
            zL=zL,
            zH=zH,
            context=context,
            l_cycles=hra_config.l_cycles,
            h_cycles=h_cycles,
        )
        
        # Return last token: [b, s, r] -> [b, r]
        return zLs[:, -1, :], zHs[:, -1, :]

    def forward(
        self,
        hidden_states: Float[Tensor, 'b s h'],
        *args: Any,
        **kwargs: Any
    ) -> Float[Tensor, 'b s h']:
        previous_dtype = hidden_states.dtype

        # Compute standard HRA output (handles disable_adapters and merged)
        result = super().forward(hidden_states, *args, **kwargs)

        if self.disable_adapters or not self.active_adapters:
            result = result.to(previous_dtype)
            return result

        # Use injected cache from Coconut.recursion_context() if available
        if self._recursion_cache is None:
            recursion_cache = {}
        else:
            recursion_cache = self._recursion_cache

        base_layer = self.get_base_layer()

        # FIXME add persistent steering as in delora
        # FIXME move to get_delta

        # Apply TRM HRA refinement delta on top of HRA output
        # HRA provides orthogonal update; TRM refines low-rank coefficients in u-basis for additive delta
        for adapter in self.active_adapters:
            if adapter not in self.hra_u:
                continue

            # Normalize u columns as in HRA non-GS case
            opt_u = self.hra_u[adapter] / self.hra_u[adapter].norm(dim=0, keepdim=True)

            # Project last token hidden to low-rank r-dim using u as basis (parallel to A @ x_last)
            context_hs = x[:, -1, :] @ opt_u  # [b, r]
            b = context_hs.shape[0]

            # Initialize or retrieve zH and zL in r_dim
            zL = recursion_cache.get('zL', None)
            if zL is None:
                zL = self.hra_zL_init[adapter].unsqueeze(0).expand(b, -1).to(result.device)
            zH = recursion_cache.get('zH', None)
            if zH is None:
                zH = self.hra_zH_init[adapter].unsqueeze(0).expand(b, -1).to(result.device)

            # Run TRM recursion: trm(context_hs, zL, zH) -> refined zL, zH
            zL, zH = self.trm(adapter, zL, zH, context_hs)

            # Reconstruct input delta in span of u: u @ zH^T .T -> [b, in_features]
            input_delta = (opt_u @ zH.T).T  # [b, in_features]

            # Project through base weight to output delta, broadcast across sequence
            delta_hidden = base_layer.weight @ input_delta.unsqueeze(1)  # [b, 1, out_features]

            # Apply LoRA-style scaling (alpha / r) to the refinement delta
            scaling = self.hra_alpha[adapter] / self.r[adapter]
            delta = delta_hidden * scaling
            
            result += delta

            # Update cache for next layer
            recursion_cache['zL'] = zL
            recursion_cache['zH'] = zH

        if self._recursion_cache is not None:
            self._recursion_cache = recursion_cache

        result = result.to(previous_dtype)
        return result

class TRMHraLinear(nn.Module, TRMHraLayer):
    """TRM HRA implemented in a dense layer"""
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        hra_config: TRMHraAConfig,
        r: int,
        apply_GS: bool,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        TRMHraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, 
            hra_config=hra_config,
            r=r,
            apply_GS=apply_GS,
            init_weights=init_weights,
            **kwargs,
        )

    def forward(self, hidden_states: Float[Tensor, 'b s h'], *args: Any, **kwargs: Any) -> Float[Tensor, 'b s h']:
        """Forward pass - delegates to TRMHraLayer.forward"""
        return TRMHraLayer.forward(self, hidden_states, *args, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Merge not implemented for TRM HRA yet")

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers"""
        raise NotImplementedError("Unmerge not implemented for TRM HRA yet")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmhra." + rep

class TRMHraModel(HRAModel):
    """
    TRM HRA Model - subclasses HRAModel to reuse all infrastructure.
    Only overrides _create_new_module to inject TRMHraLinear instead of standard HRA.
    """

    @staticmethod
    def _create_new_module(hra_config, adapter_name, target, **kwargs):
        """
        Override to create TRMHraLinear instead of standard HRA layers.
        Falls back to parent implementation for non-TRM configs.
        """
        # Check if this is a TRM config
        if not isinstance(hra_config, TRMHraAConfig):
            # Not a TRM config, use parent's implementation
            return HRAModel._create_new_module(hra_config, adapter_name, target, **kwargs)
        
        # TRM-specific creation
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            r = hra_config.r
            apply_GS = hra_config.apply_GS
            init_weights = hra_config.init_weights
            new_module = TRMHraLinear(
                target, 
                adapter_name, 
                hra_config=hra_config,
                r=r,
                apply_GS=apply_GS,
                init_weights=init_weights,
                **kwargs
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported for TRM HRA. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module
