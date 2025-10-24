"""
This subclasses lora, you can see AdaLora for an example subclassing lora in peft
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from jaxtyping import Float

from peft.utils import PeftType
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.tuners.lora import LoraLayer
from peft.tuners.lora.config import LoraConfig
from peft.tuners._buffer_dict import BufferDict
from peft.utils.other import get_pattern_key

from .trm_adapter import L_net






@dataclass
class TRMConfig(LoraConfig):
    """
    Configuration for TRM LoRA adapter.
    Inherits from LoraConfig to get all standard LoRA features:
    - init_lora_weights strategies (gaussian, pissa, loftq, olora, eva, corda, orthogonal)
    - loftq_config, eva_config, corda_config
    - use_rslora, use_dora
    - rank_pattern, alpha_pattern
    - All other LoRA config options
    """
    # Override defaults for TRM
    use_rslora: bool = field(default=True, metadata={"help": "Use rank-stabilized LoRA (recommended for stability)"})
    
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
        # Override peft_type for TRM
        try:
            self.peft_type = PeftType.TRMLORA
        except AttributeError:
            self.peft_type = "TRMLORA"
        
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
        "l_nets",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "lora_alpha",
        "scaling",
        "lora_dropout",
        "zL_init",
        "zH_init",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__(base_layer, **kwargs)
        # TRM-specific state
        self.zL_init = BufferDict({})
        self.zH_init = BufferDict({})
        self.trm_configs: Dict[str, TRMConfig] = {}
        self.l_nets = nn.ModuleDict({})
        
        # Marker for Coconut to find TRM layers
        self._recursion_cache = None  # Injected by Coconut.recursion_context()
    
    def update_layer(
        self,
        adapter_name: str,
        trm_config: TRMConfig,
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
        
        self.trm_configs[adapter_name] = trm_config

        self.l_nets[adapter_name] = L_net(
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
        self.zL_init[adapter_name] = zL
        self.zH_init[adapter_name] = zH

    def trm(self, adapter_name: str, zL: Float[Tensor, 'b h'], zH: Float[Tensor, 'b h'], context_hs: Float[Tensor, 'b h']) -> tuple[Float[Tensor, 'b h'], Float[Tensor, 'b h']]:
        """
        Tiny Recursion Module (TRM) adapted from trm_adapter.py.
        
        Gradient flow: Early H cycles run no_grad (detached), final cycles keep grad.
        When added to base_hidden (which has grad), detached recursions act as leaf nodes,
        allowing model to learn error cleanup from its own accumulated mistakes (see TRM paper).
        """
        trm_config = self.trm_configs[adapter_name]
        l_net = self.l_nets[adapter_name]
        
        # Expect zL, zH to be [b, h]
        zLs = zL.unsqueeze(1)  # [b, 1, h]
        zHs = zH.unsqueeze(1)  # [b, 1, h]

        # Early H cycles detached: forms leaf nodes but gradients still flow via base_hidden trunk
        with torch.no_grad():
            for _ in range(max(0, trm_config.h_cycles - 1)):
                for _ in range(trm_config.l_cycles):
                    zLs = l_net(zLs, zHs + context_hs.unsqueeze(1))  # context_hs is now [b, r]
                zHs = l_net(zHs, zLs)

        # Last H cycle with grad for the final pass
        for _ in range(trm_config.l_cycles):
            zLs = l_net(zLs, zHs + context_hs.unsqueeze(1))
        zHs = l_net(zHs, zLs)

        # # FIXME is this right
        # # Additional refinement cycles (with grad)
        # for _ in range(trm_config.cycles):
        #     for _ in range(trm_config.l_cycles):
        #         zLs = l_net(zLs, zHs + context_hs.unsqueeze(1))
        #     zHs = l_net(zHs, zLs)

        # Return (zL_next, zH_next) to match downstream expectation
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

            # Run base layer
            base_hidden = self.base_layer(hidden_states, *args, **kwargs)
            result = base_hidden

            # Apply TRM LoRA adapters
            # lora $ h = W @ x + B @ (A @ x) $
            for adapter in self.active_adapters:
                if adapter not in self.lora_B:
                    continue

                # Use input hidden states (not output) for lora_A projection
                hs = hidden_states[:, -1, :]
                b = hs.shape[0]

                # Initialize zH and zL in r_dim
                zL = recursion_cache.get('zL', None)
                if zL is None:
                    zL = self.zL_init[adapter].unsqueeze(0).expand(b, -1).to(base_hidden.device)
                zH = recursion_cache.get('zH', None)
                if zH is None:
                    zH = self.zH_init[adapter].unsqueeze(0).expand(b, -1).to(base_hidden.device)
                
                # Project context to low-rank dim via lora_A
                x_down = self.lora_A[adapter](hs)  # [b, r]
                x_down = self.lora_dropout[adapter](x_down)

                # Run TRM recursion in low-rank space
                zL, zH = self.trm(adapter, zL, zH, x_down)

                # Up-project refined state via lora_B with standard LoRA scaling
                delta = self.lora_B[adapter](zH) * self.scaling[adapter]  # [b, out_features]
        
                # Add to base output (broadcast across sequence)
                result = result + delta.unsqueeze(1)  # [b, 1, out] → [b, s, out]

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
        trm_config: TRMConfig,
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

from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
TRANSFORMERS_MODELS_TO_TRMLORA_TARGET_MODULES_MAPPING = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.copy()

class TRMLoraModel(BaseTuner):
    """
    TRM LoRA Model that uses TRMLoraLayer as the tuner layer class.
    Proper PEFT model inheriting from BaseTuner.
    """
    prefix: str = "trmlora_"
    tuner_layer_cls = TRMLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_TRMLORA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        config,  # PEFT passes this as first arg
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # config is our TRMConfig
        trm_config = config

        # Regexp matching for patterns (like LoraModel does)
        r_key = get_pattern_key(trm_config.rank_pattern.keys(), current_key) if trm_config.rank_pattern else None
        r = trm_config.rank_pattern.get(r_key, trm_config.r) if r_key else trm_config.r
        
        alpha_key = get_pattern_key(trm_config.alpha_pattern.keys(), current_key) if trm_config.alpha_pattern else None
        alpha = trm_config.alpha_pattern.get(alpha_key, trm_config.lora_alpha) if alpha_key else trm_config.lora_alpha

        # Build kwargs for LoRA params
        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": trm_config.lora_dropout,
            "init_lora_weights": trm_config.init_lora_weights,
            "use_rslora": trm_config.use_rslora,
            "use_dora": trm_config.use_dora,
        }

        if isinstance(target, TRMLoraLayer):
            target.update_layer(adapter_name, trm_config=trm_config, **kwargs)
        else:
            new_module = self._create_new_module(trm_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(trm_config, adapter_name, target, **kwargs):
        """Create new TRM LoRA module - follows PEFT pattern"""
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = TRMLinear(target, adapter_name, trm_config=trm_config, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` is supported."
            )
        return new_module


