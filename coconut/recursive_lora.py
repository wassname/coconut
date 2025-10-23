import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from jaxtyping import Float

from peft.config import PeftConfig
from peft.utils import PeftType
from peft.tuners.tuners_utils import BaseTunerLayer, BaseTuner, check_target_module_exists
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
        # Use PeftType enum - register custom type if needed
        try:
            self.peft_type = PeftType.TRMLORA
        except AttributeError:
            # Fallback if TRMLORA not in enum yet
            self.peft_type = "TRMLORA"
        if isinstance(self.target_modules, list):
            self.target_modules = set(self.target_modules)

class TRMLoraLayer(BaseTunerLayer):
    """
    TRM LoRA layer that wraps a base layer and overrides forward for inline recursion and low-rank delta.
    Proper PEFT layer inheriting from BaseTunerLayer.
    """
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = (
        "trmlora_A",
        "trmlora_B",
    )
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "r",
        "trmlora_dropout",
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.r = {}
        self.trmlora_dropout = nn.ModuleDict({})
        self.trmlora_A = nn.ParameterDict({})
        self.trmlora_B = nn.ParameterDict({})
        
        # Mark the weight as unmerged (PEFT pattern uses underscore)
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs
        
        # Get base layer info
        base_layer_mod = self.get_base_layer()
        if isinstance(base_layer_mod, nn.Linear):
            self.in_features, self.out_features = base_layer_mod.in_features, base_layer_mod.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer_mod)}")
    
    @property
    def merged(self) -> bool:
        """Check if any adapters are merged"""
        return bool(self.merged_adapters)
    
    @property
    def disable_adapters(self) -> bool:
        """Property to access disable_adapters state"""
        return self._disable_adapters
    
    @disable_adapters.setter
    def disable_adapters(self, value: bool) -> None:
        """Property setter for disable_adapters"""
        self._disable_adapters = value


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
        self.trmlora_A[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))
        self.trmlora_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        
        if lora_dropout > 0.0:
            module_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            module_dropout_layer = nn.Identity()
        self.trmlora_dropout.update(nn.ModuleDict({adapter_name: module_dropout_layer}))

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
            nn.init.kaiming_uniform_(self.trmlora_A[adapter_name], a=5**0.5)
            nn.init.zeros_(self.trmlora_B[adapter_name])

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
                if adapter not in self.trmlora_A:
                    continue

                trm_config = self.trm_configs[adapter]
                
                # Get batch size from base_hidden
                if base_hidden.dim() == 2:
                    b = 1  # Assume batch size 1 for 2D tensors
                    context_hs = base_hidden.mean(dim=0, keepdim=True)  # [1, h]
                else:
                    b = base_hidden.shape[0]
                    # Pool context hidden states (use mean pooling)
                    context_hs = base_hidden.mean(dim=1) if base_hidden.dim() == 3 else base_hidden.mean(dim=0, keepdim=True)

                # Initialize zH and zL
                zH = getattr(self, f"zH_init_{adapter}").unsqueeze(0).expand(b, -1).to(base_hidden.device)
                zL = getattr(self, f"zL_init_{adapter}").unsqueeze(0).expand(b, -1).to(base_hidden.device)

                # Project context from LLM space to TRM space using transcoder
                # Transcoder goes: LLM hidden -> TRM hidden
                context_trm = self.transcoders[adapter](context_hs)  # [b, trm_hidden]
                
                # Run HRM recursion: returns (zL_next, zH_next)
                zL_next, zH_next = self.hrm(adapter, zL, zH, context_trm)

                # zH_next is already in TRM space, use it directly as features
                features = zH_next

                # Standard LoRA forward: x @ A.T @ B.T
                dropout_x = self.trmlora_dropout[adapter](features)
                delta_weight = self.trmlora_B[adapter] @ self.trmlora_A[adapter]  # [out, in]
                delta_weight = delta_weight * (trm_config.lora_alpha / max(1, self.r[adapter]))
                
                # Apply LoRA delta as weight modification, cast to match input dtype
                delta = nn.functional.linear(base_hidden, delta_weight.to(base_hidden.dtype))
                result = result + delta

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

    def forward(self, hidden_states: Float[Tensor, 'b s h'], *args: Any, **kwargs: Any) -> Float[Tensor, 'b s h']:
        """Forward pass - delegates to TRMLoraLayer.forward"""
        return TRMLoraLayer.forward(self, hidden_states, *args, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge adapter weights into base layer"""
        from peft.tuners.tuners_utils import check_adapters_to_merge
        
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.trmlora_A.keys():
                base_layer = self.get_base_layer()
                # Compute delta weight
                A = self.trmlora_A[active_adapter]
                B = self.trmlora_B[active_adapter]
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
                # Note: merged property is computed from merged_adapters list

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers"""
        import warnings
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.trmlora_A.keys():
                base_layer = self.get_base_layer()
                A = self.trmlora_A[active_adapter]
                B = self.trmlora_B[active_adapter]
                trm_config = self.trm_configs[active_adapter]
                delta_weight = (B @ A) * (trm_config.lora_alpha / max(1, self.r[active_adapter]))
                base_layer.weight.data -= delta_weight
        # Note: merged property is computed from merged_adapters list

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "trmlora." + rep

from peft.utils.constants import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
TRANSFORMERS_MODELS_TO_TRMLORA_TARGET_MODULES_MAPPING = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.copy()

class TRMModel(BaseTuner):
    """
    TRM LoRA Model that uses TRMLoraLayer as the tuner layer class.
    Proper PEFT model inheriting from BaseTuner.
    """
    prefix: str = "trmlora_"
    tuner_layer_cls = TRMLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_TRMLORA_TARGET_MODULES_MAPPING

    def _check_new_adapter_config(self, config: TRMConfig) -> None:
        """Check config when adding new adapter"""
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(trm_config, key):
        """Check if key matches target modules in config"""
        return check_target_module_exists(trm_config, key)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """Prepare adapter config, setting target_modules if not specified"""
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_TRMLORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_TRMLORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Mark only adapter parameters as trainable"""
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "trmlora_only":
                for name, m in model.named_modules():
                    if isinstance(m, TRMLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def _set_adapter_layers(self, enabled=True):
        """Enable or disable adapter layers"""
        from peft.utils import ModulesToSaveWrapper
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        """Enable all adapters"""
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        """Disable all adapters"""
        import warnings
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        """Set the active adapter"""
        import warnings
        for module in self.model.modules():
            if isinstance(module, TRMLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "base_model":
                raise
            return getattr(self.model, name)

    def _replace_module(self, parent, child_name, new_module, child):
        """Replace child module with new_module in parent"""
        setattr(parent, child_name, new_module)
        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer
        
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

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



# replace ENUM with extended version, we need to replace
import peft.utils.peft_types
import enum

class PeftType2(str, enum.Enum):
    TRMLORA = 'TRMLORA'
peft.utils.peft_types.PeftType = PeftType2

register_peft_method(name="trmlora", model_cls=TRMModel, config_cls=TRMConfig)


# # Helper function to load the model with TRMConfig
# def get_trm_model(base_model, config: TRMConfig, adapter_name: str = "default"):
#     """
#     Convenience function to load a PEFT model with TRM adapter.
#     """
#     from peft import get_peft_model
#     return get_peft_model(base_model, config, adapter_name=adapter_name)
