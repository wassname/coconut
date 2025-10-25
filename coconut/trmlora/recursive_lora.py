"""
This subclasses lora, you can see AdaLora for an example subclassing lora in peft
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from jaxtyping import Float

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.tuners.lora import LoraLayer
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.config import LoraConfig
from peft.tuners._buffer_dict import BufferDict

from .trm_adapter import L_net






@dataclass
class TRMLoraAConfig(LoraConfig):
    """
    Configuration for TRM LoRA adapter.
    Inherits from LoraConfig to get all standard LoRA features:
    - init_lora_weights strategies (gaussian, pissa, loftq, olora, eva, corda, orthogonal)
    - loftq_config, eva_config, corda_config
    - use_rslora, use_dora
    - rank_pattern, alpha_pattern
    - All other LoRA config options
    """
    # Override LoRA defaults for stability
    use_rslora: bool = field(default=True, metadata={"help": "Use rank-stabilized LoRA (recommended for stability)"})
    init_lora_weights: str = field(default="gaussian", metadata={"help": "Simple stable initialization"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Start without dropout for simplicity"})
    use_dora: bool = field(default=False, metadata={"help": "Disable DoRA to reduce complexity"})
    r: int = field(default=8, metadata={"help": "Low rank - start small"})
    lora_alpha: int = field(default=16, metadata={"help": "2*r is common with rslora"})
    
    # TRM-specific parameters
    l_cycles: int = field(default=6, metadata={"help": "Number of L_net cycles per H cycle"})
    h_cycles: int = field(default=2, metadata={"help": "Number of H cycles"})
    expansion: float = field(default=2.67, metadata={"help": "TRM expansion factor"})
    l_layers: int = field(default=2, metadata={"help": "Number of L_net layers"})
    num_heads: int = field(default=8, metadata={"help": "Number of attention heads"})
    update_mode: str = field(default="lora", metadata={"help": "Update mode: 'lora' or 'add_dora'"})
    transcoder_layers: int = field(default=2, metadata={"help": "Number of transcoder layers"})
    cycles: int = field(default=1, metadata={"help": "Additional refinement cycles"})

    def __post_init__(self):
        super().__post_init__()
        # Keep peft_type as LORA since we're just extending LoRA
        # TRMLoraModel will handle TRM-specific behavior
        
        # Validate TRM-specific constraints
        if self.h_cycles < 1:
            raise ValueError(f"h_cycles must be >= 1, got {self.h_cycles}")
        if self.l_cycles < 1:
            raise ValueError(f"l_cycles must be >= 1, got {self.l_cycles}")

class TRMLoraLayer(LoraLayer):
    """
    TRM LoRA layer that wraps a base layer and overrides forward for inline recursion and low-rank delta.
    Subclasses LoraLayer for proper PEFT integration (following AdaLora pattern).
    """
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "lora_A",
        "lora_B",
        "lora_l_nets",
        "lora_zL_init",
        "lora_zH_init",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "lora_alpha",
        "scaling",
        "lora_dropout",
        'lora_configs',
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer, **kwargs)
        # TRM-specific state, need to prefix with lora for saving
        self.lora_zL_init = BufferDict({})
        self.lora_zH_init = BufferDict({})
        self.lora_l_nets = nn.ModuleDict({})
        self.lora_configs: Dict[str, TRMLoraAConfig] = {}
        
        # Marker for Coconut to find TRM layers
        self._recursion_cache = None  # Injected by Coconut.recursion_context()
    
    def update_layer(
        self,
        adapter_name: str,
        trm_config: TRMLoraAConfig,
        **kwargs
    ) -> None:
        """
        Extend LoraLayer.update_layer to add TRM-specific components.
        Calls parent to handle standard LoRA setup (A, B, scaling, dropout, DoRA, etc).
        All LoRA params (r, lora_alpha, init_lora_weights, use_rslora, use_dora, etc) passed via kwargs.
        """
        # Call parent to handle standard LoRA setup (supports DoRA, rsLoRA, etc)
        super().update_layer(
            adapter_name=adapter_name,
            **kwargs
        )
        
        # Add TRM-specific components
        r = kwargs.get('r', trm_config.r)
            
        base_weight = self.get_base_layer().weight
        device = base_weight.device
        
        self.lora_configs[adapter_name] = trm_config

        self.lora_l_nets[adapter_name] = L_net(
            r,
            trm_config.l_layers,
            trm_config.num_heads,
            trm_config.expansion,
        )#.to(device)

        # Initialize TRM recursion states in r_dim
        zH = torch.empty(r, device=device)
        zL = torch.empty(r, device=device)
        torch.nn.init.trunc_normal_(zH, std=1.0)
        torch.nn.init.trunc_normal_(zL, std=1.0)
        self.lora_zL_init[adapter_name] = zL
        self.lora_zH_init[adapter_name] = zH

    def trm(self, adapter_name: str, zL: Float[Tensor, 'b h'], zH: Float[Tensor, 'b h'], context_hs: Float[Tensor, 'b h']) -> tuple[Float[Tensor, 'b h'], Float[Tensor, 'b h']]:
        """
        Tiny Recursion Module (TRM) adapted from trm_adapter.py.
        
        Gradient flow: Early H cycles run no_grad (detached), final cycles keep grad.
        When added to base_hidden (which has grad), detached recursions act as leaf nodes,
        allowing model to learn error cleanup from its own accumulated mistakes (see TRM paper).
        """
        trm_config = self.lora_configs[adapter_name]
        l_net = self.lora_l_nets[adapter_name]
        
        # Expect zL, zH to be [b, h]
        zLs = zL.unsqueeze(1)  # [b, 1, h]
        zHs = zH.unsqueeze(1)  # [b, 1, h]
        context = context_hs.unsqueeze(1)  # [b, 1, h]

        # Early H cycles detached: forms leaf nodes but gradients still flow via base_hidden trunk and also via `context`
        with torch.no_grad():
            for _ in range(max(0, trm_config.h_cycles - 1)):
                # L cycles: refine zL with context + zH injection
                for _ in range(trm_config.l_cycles):
                    zLs = l_net(zLs, context + zHs)
                # H cycle: refine zH with zL injection
                zHs = l_net(zHs, zLs)

        # Last H cycle with grad for backprop
        for _ in range(trm_config.l_cycles):
            zLs = l_net(zLs, context + zHs)
        zHs = l_net(zHs, zLs)

        # Return (zL_next, zH_next) in [b, h]
        return zLs.squeeze(1), zHs.squeeze(1)

    def forward(
        self,
        hidden_states: Float[Tensor, 'b s h'],
        *args: Any,
        **kwargs: Any
    ) -> Float[Tensor, 'b s h']:
        # Use injected cache from Coconut.recursion_context() if available
        if self._recursion_cache is None:
            recursion_cache = {}
        else:
            recursion_cache = self._recursion_cache

        previous_dtype = hidden_states.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(hidden_states, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(hidden_states, *args, **kwargs)
        else:
            if not self.active_adapters:
                return self.base_layer(hidden_states, *args, **kwargs).to(previous_dtype)

            # Run base layer: W @ x
            base_hidden = self.base_layer(hidden_states, *args, **kwargs)
            result = base_hidden

            # Apply TRM LoRA adapters
            # Standard LoRA: h = W @ x + B @ (A @ x)
            # TRM LoRA:      h = W @ x + B @ trm(A @ x)
            for adapter in self.active_adapters:
                if adapter not in self.lora_B:
                    continue

                # Project INPUT (not output) down to low-rank via lora_A
                # Standard LoRA uses the layer input, not output
                x_down = self.lora_A[adapter](self.lora_dropout[adapter](hidden_states))  # [b, s, r]
                # NOTE lora_a never gets grad, so it 
                
                # For TRM, use last token's projection as context
                context_hs = x_down[:, -1, :]  # [b, r]
                b = context_hs.shape[0]

                # Initialize or retrieve zH and zL in r_dim
                zL = recursion_cache.get('zL', None)
                if zL is None:
                    zL = self.lora_zL_init[adapter].unsqueeze(0).expand(b, -1).to(base_hidden.device)
                zH = recursion_cache.get('zH', None)
                if zH is None:
                    zH = self.lora_zH_init[adapter].unsqueeze(0).expand(b, -1).to(base_hidden.device)

                # Run TRM recursion: trm(A @ x, zL, zH)
                zL, zH = self.trm(adapter, zL, zH, context_hs)

                # Up-project refined state via lora_B with standard LoRA scaling
                delta = self.lora_B[adapter](zH) * self.scaling[adapter]  # [b, out_features]
        
                # Add to base output (broadcast across sequence)
                result = result + delta.unsqueeze(1)  # [b, 1, out] → [b, s, out]

                # Update cache for next layer
                recursion_cache['zL'] = zL
                recursion_cache['zH'] = zH

        result = result.to(previous_dtype)
        return result

class TRMLinear(nn.Module, TRMLoraLayer):
    """TRM LoRA implemented in a dense layer"""
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        trm_config: TRMLoraAConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        TRMLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, 
            trm_config=trm_config,
            **kwargs,
        )

    def forward(self, hidden_states: Float[Tensor, 'b s h'], *args: Any, **kwargs: Any) -> Float[Tensor, 'b s h']:
        """Forward pass - delegates to TRMLoraLayer.forward"""
        return TRMLoraLayer.forward(self, hidden_states, *args, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Merge not implemented for TRM LoRA yet")

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers"""
        raise NotImplementedError("Unmerge not implemented for TRM LoRA yet")

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmlora." + rep

class TRMLoraModel(LoraModel):
    """
    TRM LoRA Model - subclasses LoraModel to reuse all infrastructure.
    Only overrides _create_new_module to inject TRMLinear instead of standard LoRA.
    """
    
    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        """
        Override to create TRMLinear instead of standard LoRA layers.
        Falls back to parent implementation for non-TRM configs.
        """
        # Check if this is a TRM config
        if not isinstance(lora_config, TRMLoraAConfig):
            # Not a TRM config, use parent's implementation
            return LoraModel._create_new_module(lora_config, adapter_name, target, **kwargs)
        
        # TRM-specific creation
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = TRMLinear(target, adapter_name, trm_config=lora_config, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported for TRM LoRA. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module


