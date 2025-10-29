"""
This subclasses delora, you can see AdaLora for an example subclassing lora in peft
Adapted from recursive_lora.py to use DeLoRA instead of LoRA, integrating TRM recursion in the scaled intermediate space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from jaxtyping import Float
from einops import repeat, rearrange

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.delora.layer import DeloraLayer
from peft.tuners.delora.model import DeloraModel
from peft.tuners.delora.config import DeloraConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils import PeftType

from .trm_adapter import L_net, trm_recursion

@dataclass
class TRMDeloraAConfig(DeloraConfig):
    """
    Configuration for TRM DeLoRA adapter.
    Inherits from DeloraConfig to get all standard DeLoRA features.
    """
    # TRM-specific parameters
    l_cycles: int = field(default=6, metadata={"help": "Number of L_net cycles per H cycle"})
    h_cycles: int = field(default=2, metadata={"help": "Number of H cycles"})
    expansion: float = field(default=2.67, metadata={"help": "TRM expansion factor"})
    l_layers: int = field(default=2, metadata={"help": "Number of L_net layers"})
    num_heads: int = field(default=8, metadata={"help": "Number of attention heads"})
    update_mode: str = field(default="delora", metadata={"help": "Update mode: 'delora' or 'add_dora'"})
    transcoder_layers: int = field(default=2, metadata={"help": "Number of transcoder layers"})
    cycles: int = field(default=1, metadata={"help": "Additional refinement cycles"})

    def __post_init__(self):
        super().__post_init__()
        # Ensure peft_type is TRMDELORA
        self.peft_type = 'TRMDELORA'
        
        # Validate TRM-specific constraints
        if self.h_cycles < 1:
            raise ValueError(f"h_cycles must be >= 1, got {self.h_cycles}")
        if self.l_cycles < 1:
            raise ValueError(f"l_cycles must be >= 1, got {self.l_cycles}")

class TRMDeloraLayer(DeloraLayer):
    """
    TRM DeLoRA layer that wraps a base layer and overrides forward for inline recursion and low-rank delta.
    Subclasses DeloraLayer for proper PEFT integration.
    """
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "delora_A",
        "delora_B",
        "delora_lambda",
        "delora_l_nets",
        "delora_output_head",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "delora_dropout",
        "delora_w_norm",
        "delora_zL_init",
        "delora_zH_init",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer, **kwargs)
        # TRM-specific state, prefixed with delora for saving
        self.delora_zL_init = BufferDict({})
        self.delora_zH_init = BufferDict({})
        self.delora_configs: Dict[str, TRMDeloraAConfig] = {}
        self.delora_l_nets = nn.ModuleDict({})
        # Per-adapter output heads for mixing zH
        self.delora_output_head = nn.ModuleDict({})
        
        # Marker for Coconut to find TRM layers
        self._recursion_cache = None  # Injected by Coconut.recursion_context()
    
    def update_layer(
        self,
        adapter_name: str,
        trm_config: TRMDeloraAConfig,
        r: int,
        delora_lambda: float,
        module_dropout: float,
        init_weights: bool = True,
        inference_mode: bool = False,
        **kwargs
    ) -> None:
        """
        Extend DeloraLayer.update_layer to add TRM-specific components.
        Calls parent to handle standard DeLoRA setup.
        """
        # Call parent to handle standard DeLoRA setup
        super().update_layer(
            adapter_name=adapter_name,
            r=r,
            delora_lambda=delora_lambda,
            module_dropout=module_dropout,
            init_weights=init_weights,
            inference_mode=inference_mode,
            **kwargs
        )
        
        # Add TRM-specific components
        self.delora_configs[adapter_name] = trm_config

        self.delora_l_nets[adapter_name] = L_net(
            r,
            trm_config.l_layers,
            trm_config.num_heads,
            trm_config.expansion,
        )

        # Initialize TRM recursion states in r_dim
        base_weight = self.get_base_layer().weight
        device = base_weight.device
        
        zH = torch.empty(r, device=device)
        zL = torch.empty(r, device=device)
        torch.nn.init.trunc_normal_(zH, std=1.0)
        torch.nn.init.trunc_normal_(zL, std=1.0)
        self.delora_zL_init[adapter_name] = zL
        self.delora_zH_init[adapter_name] = zH

        # Initialize output head for zH
        self.delora_output_head[adapter_name] = nn.Linear(r, r, bias=False)
        nn.init.trunc_normal_(self.delora_output_head[adapter_name].weight, std=0.02)

    def trm(self, adapter_name: str, zL, zH, context_hs, h_cycles=None):
        """Wrapper around trm_recursion with adapter-specific config."""
        trm_config = self.delora_configs[adapter_name]
        if h_cycles is None:
            h_cycles = trm_config.h_cycles
        
        zLs, zHs = trm_recursion(
            l_net=self.delora_l_nets[adapter_name],
            zL=zL,
            zH=zH,
            context=context_hs,
            l_cycles=trm_config.l_cycles,
            h_cycles=h_cycles,
        )
        zHs = self.delora_output_head[adapter_name](zHs)
        return zLs, zHs


    def forward(self, x: Float[Tensor, 'b s h'], *args: Any, **kwargs: Any) -> Float[Tensor, 'b s h']:
        previous_dtype = x.dtype
        # Use injected cache from Coconut.recursion_context() if available
        # FIXME, to be consistent should this be per-adapter?
        assert len(self.active_adapters) <= 1, "TRM DeLoRA currently supports only one active adapter at a time."
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
                if adapter not in self.delora_A:
                    continue

                x_d = self.delora_dropout[adapter](x)
                
                # 1. Down-project via A: (x * w_norm) @ A.T
                h = F.linear(x_d * self.delora_w_norm[adapter], self.delora_A[adapter])  # [b, s, r]

                # 2. Normalize by A (remove A's magnitude, get unit directions, per row)
                An = torch.clamp(self.delora_A[adapter].norm(dim=1), min=1e-4)  # [r]
                h_normalized = h / An.unsqueeze(0).unsqueeze(0)  # [b, s, r] - unit norm per component
                    
                # Check if we're in steering mode (post-latent)
                steering_mode = recursion_cache.get('steering_mode', False)
                if steering_mode:
                    # Don't run TRM, just apply cached zH
                    zH = recursion_cache.get('zH')
                    zL = recursion_cache.get('zL')
                    
                    # Apply steering (detached, no grad)
                    # Bn = torch.clamp(self.delora_B[adapter].norm(dim=0), min=1e-4)
                    # scaling = (self.delora_lambda[adapter] / self.r[adapter]) / Bn
                    # h = (zH * scaling).detach()  # Detach to prevent grad flow

                    # # fold sequence dimension into b, then back out
                    # context = rearrange(h_normalized, 'b s r -> (b s) r')  # [b*s, r]
                    # zL = repeat(zL, 'b r -> (b s) r', b=h.shape[0], s=h.shape[1])
                    # zH = repeat(zH, 'b r -> (b s) r', b=h.shape[0], s=h.shape[1])

                    # TRM refines direction (operates on normalized space)
                    zLs, zHs = self.trm(adapter, zL, zH, h_normalized, h_cycles=1)  # zH is refined 1 time
                else:                       
                    """
                    TRM DeLoRA combines DeLoRA's magnitude decoupling with TRM's recursive refinement:
                    
                    DeLoRA philosophy (from paper Section 2.2):
                    - Normalize low-rank components to unit norm → learn pure directions (angles)
                    - Apply learned scaling λ separately → control adaptation strength (magnitude)
                    - This decouples angular learning from magnitude, preventing catastrophic overwriting
                    
                    TRM integration:
                    - Down-project via A to low-rank space (r-dimensional)
                    - Normalize by ||A|| to remove magnitude → get unit directions
                    - TRM recursively refines these directions (operates on normalized space)
                    - Apply λ/r/||B|| scaling to refined directions → controlled magnitude
                    - Up-project via B back to full space
                    
                    Key insight: TRM learns to refine DIRECTIONS in normalized r-space, while λ 
                    controls the final MAGNITUDE. This preserves DeLoRA's robustness properties 
                    while adding TRM's recursive reasoning capability.
                    """


                    # 3. TRM recursion on normalized directions (last token)
                    context = h_normalized#[:, -1:, :]  # [b, 1, r] - normalized direction
                    b = context.shape[0]

                    # Initialize or retrieve zH and zL in r_dim
                    zL = recursion_cache.get('zL', None)
                    if zL is None:
                        zL = self.delora_zL_init[adapter].unsqueeze(0).expand(b, -1).to(h.device)
                    zH = recursion_cache.get('zH', None)
                    if zH is None:
                        zH = self.delora_zH_init[adapter].unsqueeze(0).expand(b, -1).to(h.device)

                    # TRM refines direction (operates on normalized space)
                    zLs, zHs = self.trm(adapter, zL, zH, context)  # zH is refined direction

                    # Update cache for next layer
                    recursion_cache['zL'] = zLs[:, -1, :]  # [b, r]
                    recursion_cache['zH'] = zHs[:, -1, :]  # [b, r]

                    # zH = zH.unsqueeze(1)  # [b, 1, r]

                # 4. Apply magnitude control (lambda/r, compensate for B, columnwise)
                Bn = torch.clamp(self.delora_B[adapter].norm(dim=0), min=1e-4)  # [r]
                scaling = (self.delora_lambda[adapter] / self.r[adapter]) / Bn  # [r]
                h = zHs * scaling  # [b, 1, r] - refined direction * controlled magnitude

                # 5. Up-project via B
                h = F.linear(h, self.delora_B[adapter])  # [b, out]

                add_out += h  # [b, 1, out] broadcasts to [b, s, out], but it's only ever one token that we are processing with <latent>, so s=1

            result = base_out + add_out.to(base_out.dtype)

        result = result.to(previous_dtype)
        return result

class TRMDeloraLinear(nn.Module, TRMDeloraLayer):
    """TRM DeLoRA implemented in a dense layer"""
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        trm_config: TRMDeloraAConfig,
        r: int,
        delora_lambda: float,
        module_dropout: float,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        TRMDeloraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, 
            trm_config=trm_config,
            r=r,
            delora_lambda=delora_lambda,
            module_dropout=module_dropout,
            init_weights=init_weights,
            **kwargs,
        )

    def forward(self, hidden_states: Float[Tensor, 'b s h'], *args: Any, **kwargs: Any) -> Float[Tensor, 'b s h']:
        """Forward pass - delegates to TRMDeloraLayer.forward"""
        return TRMDeloraLayer.forward(self, hidden_states, *args, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Merge not implemented for TRM DeLoRA yet")

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers"""
        raise NotImplementedError("Unmerge not implemented for TRM DeLoRA yet")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmdelora." + rep

class TRMDeloraModel(DeloraModel):
    """
    TRM DeLoRA Model - subclasses DeloraModel to reuse all infrastructure.
    Only overrides _create_new_module to inject TRMDeloraLinear instead of standard DeLoRA.
    """
    
    @staticmethod
    def _create_new_module(delora_config, adapter_name, target, **kwargs):
        """
        Override to create TRMDeloraLinear instead of standard DeLoRA layers.
        Falls back to parent implementation for non-TRM configs.
        """
        # Check if this is a TRM config
        if not isinstance(delora_config, TRMDeloraAConfig):
            # Not a TRM config, use parent's implementation
            return DeloraModel._create_new_module(delora_config, adapter_name, target, **kwargs)
        
        # TRM-specific creation
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            # Pass required DeLoRA params
            r = kwargs.pop('r', delora_config.r)
            delora_lambda = kwargs.pop('delora_lambda', delora_config.delora_lambda)
            module_dropout = kwargs.pop('module_dropout', delora_config.module_dropout)
            init_weights = kwargs.pop('init_weights', delora_config.init_weights)
            new_module = TRMDeloraLinear(
                target, 
                adapter_name, 
                trm_config=delora_config,
                r=r,
                delora_lambda=delora_lambda,
                module_dropout=module_dropout,
                init_weights=init_weights,
                **kwargs
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported for TRM DeLoRA. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module
