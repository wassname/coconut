import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from jaxtyping import Float

from peft.config import PeftConfig
from peft.utils import PeftType
from peft.tuners.tuners_utils import BaseTunerLayer, BaseTuner
from peft.tuners._buffer_dict import BufferDict
from peft.utils.other import get_pattern_key

from .trm_adapter import L_net, TRMTranscoder, rms_norm  # Assuming rms_norm is available; adjust if needed

from peft.utils import register_peft_method


@dataclass
class TRMConfig(PeftConfig):
    """
    Configuration for TRM LoRA adapter.
    Proper PEFT config inheriting from PeftConfig.
    """
    # Basic LoRA parameters
    r: int = field(default=8, metadata={"help": "TRM LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA scaling parameter"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout probability"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression to replace with TRM LoRA."}
    )
    exclude_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "List of module names to exclude from TRM LoRA."}
    )
    bias: str = field(default="none", metadata={"help": "Bias type for TRM LoRA. Can be 'none' or 'all'"})
    
    # TRM-specific parameters
    l_cycles: int = field(default=6, metadata={"help": "Number of L_net cycles per H cycle"})
    h_cycles: int = field(default=2, metadata={"help": "Number of H cycles"})
    hidden_size: int = field(default=2048, metadata={"help": "TRM hidden size"})
    llm_hidden_size: int = field(default=2048, metadata={"help": "LLM hidden size"})
    expansion: float = field(default=2.67, metadata={"help": "TRM expansion factor"})
    l_layers: int = field(default=2, metadata={"help": "Number of L_net layers"})
    num_heads: int = field(default=8, metadata={"help": "Number of attention heads"})
    update_mode: str = field(default="lora", metadata={"help": "Update mode: 'lora' or 'add_dora'"})
    transcoder_layers: int = field(default=2, metadata={"help": "Number of transcoder layers"})
    cycles: int = field(default=1, metadata={"help": "Additional refinement cycles"})
    
    # PEFT compatibility parameters
    init_weights: bool = field(default=True, metadata={"help": "Whether to initialize weights"})
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None, metadata={"help": "Layer indices to transform"}
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None, metadata={"help": "Layer pattern name"}
    )
    rank_pattern: Optional[dict] = field(default_factory=dict, metadata={"help": "Rank pattern mapping"})
    modules_to_save: Optional[List[str]] = field(default=None, metadata={"help": "Modules to save"})

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "TRM"  # Custom TRM type
        if isinstance(self.target_modules, list):
            self.target_modules = set(self.target_modules)

class TRMLoraLayer(BaseTunerLayer):
    """
    TRM LoRA layer that wraps a base layer and overrides forward for inline recursion and low-rank delta.
    Proper PEFT layer inheriting from BaseTunerLayer.
    """
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "trm_lora_A",
        "trm_lora_B", 
        "zH_init",
        "zL_init",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "trm_lora_dropout",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.r = {}
        self.trm_lora_dropout = nn.ModuleDict({})
        self.trm_lora_A = nn.ParameterDict({})
        self.trm_lora_B = nn.ParameterDict({})
        
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs
        
        # Get base layer info
        base_layer_mod = self.get_base_layer()
        if isinstance(base_layer_mod, nn.Linear):
            self.in_features, self.out_features = base_layer_mod.in_features, base_layer_mod.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer_mod)}")


    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        trm_config: TRMConfig,
        init_weights: bool = True,
        **kwargs
    ) -> None:
        """Internal function to create TRM LoRA adapter"""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.trm_lora_A[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))
        self.trm_lora_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        
        if lora_dropout > 0.0:
            module_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            module_dropout_layer = nn.Identity()
        self.trm_lora_dropout.update(nn.ModuleDict({adapter_name: module_dropout_layer}))

        # Initialize TRM components for this adapter
        if not hasattr(self, 'trm_configs'):
            self.trm_configs = {}
        self.trm_configs[adapter_name] = trm_config
        
        # Initialize TRM networks
        if not hasattr(self, 'l_nets'):
            self.l_nets = {}
        if not hasattr(self, 'transcoders'):
            self.transcoders = {}
            
        self.l_nets[adapter_name] = L_net(
            trm_config.hidden_size,
            trm_config.llm_hidden_size,
            trm_config.l_layers,
            trm_config.num_heads,
            trm_config.expansion,
        )

        self.transcoders[adapter_name] = TRMTranscoder(
            trm_config.hidden_size,
            trm_config.llm_hidden_size,
            expansion=trm_config.expansion,
            trm_transcoder_layers=trm_config.transcoder_layers,
        )
        self.transcoders[adapter_name].final_proj = nn.Identity()

        # Initialize initial states
        zH = torch.empty(trm_config.hidden_size)
        zL = torch.empty(trm_config.hidden_size)
        torch.nn.init.trunc_normal_(zH, std=1.0)
        torch.nn.init.trunc_normal_(zL, std=1.0)
        self.register_buffer(f"zH_init_{adapter_name}", zH, persistent=True)
        self.register_buffer(f"zL_init_{adapter_name}", zL, persistent=True)

        # Initialize LoRA weights
        if init_weights:
            nn.init.kaiming_uniform_(self.trm_lora_A[adapter_name], a=5**0.5)
            nn.init.zeros_(self.trm_lora_B[adapter_name])

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def hrm(self, adapter_name: str, zL: Float[Tensor, 'b h'], zH: Float[Tensor, 'b h'], context_hs: Float[Tensor, 'b h']) -> tuple[Float[Tensor, 'b h'], Float[Tensor, 'b h']]:
        """
        Hierarchical Recursion Module (HRM) adapted from trm_adapter.py.
        Runs fixed cycles with L_net.
        """
        trm_config = self.trm_configs[adapter_name]
        l_net = self.l_nets[adapter_name]
        
        # Expect zL, zH to be [b, h]
        zLs = zL.unsqueeze(1)  # [b, 1, h]
        zHs = zH.unsqueeze(1)  # [b, 1, h]

        # H cycles: run H cycles where earlier H cycles are run without grad except last
        with torch.no_grad():
            for _ in range(max(0, trm_config.h_cycles - 1)):
                for _ in range(trm_config.l_cycles):
                    zLs = l_net(zLs, zHs + context_hs.unsqueeze(1))
                zHs = l_net(zHs, zLs)

        # Last H cycle with grad for the final pass
        for _ in range(trm_config.l_cycles):
            zLs = l_net(zLs, zHs + context_hs.unsqueeze(1))
        zHs = l_net(zHs, zLs)

        # Additional refinement cycles
        for _ in range(trm_config.cycles):
            for _ in range(trm_config.l_cycles):
                zLs = l_net(zLs, zHs + context_hs.unsqueeze(1))
            zHs = l_net(zHs, zLs)

        # Return (zL_next, zH_next) to match downstream expectation
        return zLs.squeeze(1), zHs.squeeze(1)

    def forward(
        self,
        hidden_states: Float[Tensor, 'b s h'],
        *args: Any,
        **kwargs: Any
    ) -> Float[Tensor, 'b s h']:
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
            for adapter in self.active_adapters:
                if adapter not in self.trm_lora_A:
                    continue

                trm_config = self.trm_configs[adapter]
                b, s, h = base_hidden.shape

                # Initialize zH and zL
                zH = getattr(self, f"zH_init_{adapter}").unsqueeze(0).expand(b, -1).to(base_hidden.device)
                zL = getattr(self, f"zL_init_{adapter}").unsqueeze(0).expand(b, -1).to(base_hidden.device)

                # Pool context hidden states (use last token as context)
                context_hs = base_hidden[:, -1, :]

                # Run HRM recursion: returns (zL_next, zH_next)
                zL_next, zH_next = self.hrm(adapter, zL, zH, context_hs)

                # Transcoder MLP to produce features
                features = self.transcoders[adapter](zH_next)

                # Standard LoRA forward: x @ A.T @ B.T
                dropout_x = self.trm_lora_dropout[adapter](features)
                delta = nn.functional.linear(dropout_x, self.trm_lora_A[adapter])
                delta = nn.functional.linear(delta, self.trm_lora_B[adapter])
                delta = delta * (trm_config.lora_alpha / max(1, self.r[adapter]))

                # Apply delta to last token hidden (latent position assumption)
                result = result * 1.0
                result[:, -1, :] += delta

        result = result.to(previous_dtype)
        return result

class TRMLinear(nn.Module, TRMLoraLayer):
    """TRM LoRA implemented in a dense layer"""
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        trm_config: TRMConfig,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        TRMLoraLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, trm_config, init_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge adapter weights into base layer"""
        from peft.tuners.tuners_utils import check_adapters_to_merge
        
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.trm_lora_A.keys():
                base_layer = self.get_base_layer()
                # Compute delta weight
                A = self.trm_lora_A[active_adapter]
                B = self.trm_lora_B[active_adapter]
                trm_config = self.trm_configs[active_adapter]
                delta_weight = (B @ A) * (trm_config.lora_alpha / max(1, self.r[active_adapter]))
                
                with torch.no_grad():
                    if safe_merge:
                        orig_weights = base_layer.weight.data.clone()
                        orig_weights = orig_weights + delta_weight
                        if not torch.isfinite(orig_weights).all():
                            raise ValueError(f"NaNs detected in merged weights for adapter {active_adapter}")
                        base_layer.weight.data = orig_weights
                    else:
                        base_layer.weight.data.add_(delta_weight)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers"""
        import warnings
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.trm_lora_A.keys():
                base_layer = self.get_base_layer()
                A = self.trm_lora_A[active_adapter]
                B = self.trm_lora_B[active_adapter]
                trm_config = self.trm_configs[active_adapter]
                delta_weight = (B @ A) * (trm_config.lora_alpha / max(1, self.r[active_adapter]))
                base_layer.weight.data -= delta_weight

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trm_lora." + rep


class TRMModel(BaseTuner):
    """
    TRM LoRA Model that uses TRMLoraLayer as the tuner layer class.
    Proper PEFT model inheriting from BaseTuner.
    """
    prefix: str = "trm_lora_"
    tuner_layer_cls = TRMLoraLayer

    def _check_new_adapter_config(self, config: TRMConfig) -> None:
        """Check config when adding new adapter"""
        super()._check_new_adapter_config(config)

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

        # Debug: print what we're getting
        print(f"DEBUG _create_and_replace: current_key = {current_key}")
        print(f"DEBUG _create_and_replace: target type = {type(target)}")
        print(f"DEBUG _create_and_replace: target_name = {target_name}")

        # Regexp matching for patterns
        r_key = get_pattern_key(trm_config.rank_pattern.keys(), current_key) if trm_config.rank_pattern else None
        r = trm_config.rank_pattern.get(r_key, trm_config.r) if r_key else trm_config.r

        if isinstance(target, TRMLoraLayer):
            target.update_layer(
                adapter_name,
                r=r,
                lora_alpha=trm_config.lora_alpha,
                lora_dropout=trm_config.lora_dropout,
                trm_config=trm_config,
                init_weights=trm_config.init_weights,
            )
        else:
            new_module = self._create_new_module(
                trm_config,
                adapter_name, 
                target, 
                r=r,
                lora_alpha=trm_config.lora_alpha,
                lora_dropout=trm_config.lora_dropout,
                init_weights=trm_config.init_weights,
            )
            if adapter_name != self.active_adapter:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(self, trm_config, adapter_name, target, **kwargs):
        # Debug: print what we're getting
        print(f"DEBUG: target type = {type(target)}")
        print(f"DEBUG: target = {target}")
        if hasattr(target, '__dict__'):
            print(f"DEBUG: target attrs = {list(target.__dict__.keys())}")
        
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
            print(f"DEBUG: base_layer type = {type(target_base_layer)}")
        else:
            target_base_layer = target
            print(f"DEBUG: using target as base_layer")

        # Support both nn.Linear and other layer types that contain linear layers
        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = TRMLinear(target, adapter_name, trm_config=trm_config, **kwargs)
        elif hasattr(target_base_layer, 'weight') and len(target_base_layer.weight.shape) == 2:
            # Treat any 2D weight layer as linear-like
            new_module = TRMLinear(target, adapter_name, trm_config=trm_config, **kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {type(target_base_layer)}")
        return new_module

    def _create_new_module(self, trm_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = TRMLinear(target, adapter_name, **kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {type(target_base_layer)}")
        return new_module

    # Required abstract methods from BaseTuner
    def _check_target_module_exists(self, trm_config: TRMConfig, key: str) -> bool:
        """Check if target modules exist"""
        # Simple implementation - assume they exist if target_modules is set
        return trm_config.target_modules is not None

    def _mark_only_adapters_as_trainable(self) -> None:
        """Mark only adapter weights as trainable"""
        for n, p in self.model.named_parameters():
            if 'trm_lora_' not in n:
                p.requires_grad = False

    def _prepare_adapter_config(self, peft_config, adapter_name: str):
        """Prepare adapter config"""
        return peft_config

    def disable_adapter_layers(self) -> None:
        """Disable adapter layers"""
        for module in self.model.modules():
            if isinstance(module, TRMLoraLayer):
                module.disable_adapters = True

    def enable_adapter_layers(self) -> None:
        """Enable adapter layers"""
        for module in self.model.modules():
            if isinstance(module, TRMLoraLayer):
                module.disable_adapters = False

    def print_trainable_parameters(self) -> None:
        """Print trainable parameters info"""
        super().print_trainable_parameters()
        if hasattr(self, 'peft_config') and self.peft_config:
            config = list(self.peft_config.values())[0]
            if hasattr(config, 'cycles'):
                print(f"TRM-specific params: cycles={config.cycles}, expansion={config.expansion}")

PeftType.TRM = PeftType('TRM', 'TRM')
register_peft_method(name="trm", model_cls=TRMModel, config_cls=TRMConfig)


# Helper function to load the model with TRMConfig
def get_trm_model(base_model, config: TRMConfig, adapter_name: str = "default"):
    """
    Convenience function to load a PEFT model with TRM adapter.
    """
    from peft import get_peft_model
    return get_peft_model(base_model, config, adapter_name=adapter_name)
