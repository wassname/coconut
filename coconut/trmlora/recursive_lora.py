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

from .trm_adapter import L_net






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
    # hidden_size: int = field(default=2048, metadata={"help": "TRM hidden size"})
    # llm_hidden_size: int = field(default=2048, metadata={"help": "LLM hidden size"})
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
        # "trmlora_A", # 
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
        # self.trmlora_A = nn.ParameterDict({}) # FIXME is it used?
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
        
        base_weight = self.get_base_layer().weight
        device = base_weight.device

        self.r[adapter_name] = r
        # Removed unused trmlora_A
        self.trmlora_B[adapter_name] = nn.Parameter(torch.empty(self.out_features, r, device=device))

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

        # Early creation of DoRA and DeLoRA params to avoid AttributeError
        if not hasattr(self, 'dora_magnitudes'):
            self.dora_magnitudes = nn.ParameterDict({})
        if adapter_name not in self.dora_magnitudes:
            self.dora_magnitudes[adapter_name] = nn.Parameter(torch.ones(self.out_features))

        if not hasattr(self, 'delora_lambda'):
            self.delora_lambda = nn.ParameterDict({})
        if adapter_name not in self.delora_lambda:
            self.delora_lambda[adapter_name] = nn.Parameter(torch.tensor(1.0))

        r_dim = self.r[adapter_name]
        r_dim = self.r[adapter_name]
        self.down_projs = nn.ModuleDict({})
        self.down_projs[adapter_name] = nn.Linear(self.out_features, r_dim)

        self.l_nets[adapter_name] = L_net(
            r_dim,
            trm_config.l_layers,
            trm_config.num_heads,
            trm_config.expansion,
        ).to(device)

        # Remove transcoder MLP, use LoRA-style up-proj with B
        # self.transcoders[adapter_name] = ... (removed)

        # Initialize initial states in r_dim
        zH = torch.empty(r_dim, device=device)
        zL = torch.empty(r_dim, device=device)
        torch.nn.init.trunc_normal_(zH, std=1.0)
        torch.nn.init.trunc_normal_(zL, std=1.0)
        self.register_buffer(f"zH_init_{adapter_name}", zH, persistent=True)
        self.register_buffer(f"zL_init_{adapter_name}", zL, persistent=True)

        # Initialize LoRA weights (only B now) with small std for stability
        if init_weights:
            nn.init.normal_(self.trmlora_B[adapter_name], mean=0.0, std=0.02)  # Small init like in transformers

        # Init DoRA magnitude to base weight norms for stability (DeLoRA inspiration)
        base_norm = torch.norm(base_weight, dim=1)  # Per output channel
        with torch.no_grad():
            self.dora_magnitudes[adapter_name].copy_(base_norm)

        # Init lambda to reasonable bound
        with torch.no_grad():
            self.delora_lambda[adapter_name].fill_(5.0)  # Example starting bound

        # Move new weights to device (only does ModuleDicts, ParameterDict, BufferDict)
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
                    zLs = l_net(zLs, zHs + context_hs.unsqueeze(1))  # context_hs is now [b, r]
                zHs = l_net(zHs, zLs)

        # Last H cycle with grad for the final pass
        for _ in range(trm_config.l_cycles):
            zLs = l_net(zLs, zHs + context_hs.unsqueeze(1))
        zHs = l_net(zHs, zLs)

        # Additional refinement cycles (with grad)
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
                if adapter not in self.trmlora_B:
                    continue

                trm_config = self.trm_configs[adapter]
                
                # Get batch size from base_hidden
                if base_hidden.dim() == 2:
                    b = 1  # Assume batch size 1 for 2D tensors
                    context_hs = base_hidden.mean(dim=0, keepdim=True)  # [1, h]
                else:
                    b = base_hidden.shape[0]
                    # Use last hidden state instead of mean pooling
                if base_hidden.dim() == 3:
                    context_hs = base_hidden[:, -1, :]
                elif base_hidden.dim() == 2:
                    context_hs = base_hidden[-1].unsqueeze(0)
                else:
                    context_hs = base_hidden.mean(dim=0 if base_hidden.dim() == 1 else 1, keepdim=True)  # Fallback

                # Project context to low-rank dim
                # Ensure context has same dtype/device as the down projection weights to avoid matmul dtype errors
                down_proj = self.down_projs[adapter]
                # Move/cast context to the down_proj's device and dtype
                context_hs_for_proj = context_hs.to(dtype=next(down_proj.parameters()).dtype, device=next(down_proj.parameters()).device)
                context_proj = down_proj(context_hs_for_proj)  # [b, r]

                # Initialize zH and zL in r_dim
                zH = getattr(self, f"zH_init_{adapter}").unsqueeze(0).expand(b, -1).to(base_hidden.device)
                zL = getattr(self, f"zL_init_{adapter}").unsqueeze(0).expand(b, -1).to(base_hidden.device)

                # Run HRM recursion in low-rank space
                zL_next, zH_next = self.hrm(adapter, zL, zH, context_proj)  # Pass projected context

                # LoRA-style up-projection with DeLoRA-inspired normalization
                scaling = trm_config.lora_alpha / max(1, self.r[adapter])
                direction = (self.trmlora_B[adapter] @ zH_next.T).T  # [b, out]

                # Normalize per output channel (column-wise like DeLoRA)
                direction_norm = torch.norm(direction, dim=0, keepdim=True) + 1e-6  # [1, out]
                direction = direction / direction_norm

                # Bound norm with learnable lambda (DeLoRA style)
                lambda_bound = self.delora_lambda[adapter].abs()  # Scalar or per-rank?
                direction = direction * lambda_bound.clamp(min=0.1, max=10.0)  # Simple bounding

                # DoRA: Scale by learned magnitude
                magnitude = self.dora_magnitudes[adapter].unsqueeze(0).expand_as(direction)  # [b, out]
                features = magnitude * direction * scaling

                # Add dropout if needed
                features = self.trmlora_dropout[adapter](features)

                # Add per-batch features as broadcast delta (like context-dependent bias)
                result = result + features.unsqueeze(1)  # [b, 1, out] broadcast to [b, s, out]

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


