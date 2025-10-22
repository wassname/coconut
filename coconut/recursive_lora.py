import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from jaxtyping import Float

from peft import LoraConfig, LoraModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer

from .trm_adapter import L_net, TRMTranscoder, rms_norm  # Assuming rms_norm is available; adjust if needed

# FIXME not how a peft config works see https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/config.py
@dataclass
class TRMConfig(LoraConfig):
    """
    Configuration for TRM LoRA adapter.
    Subclass of LoraConfig with additional TRM-specific parameters.
    """
    def __init__(
        self,
        l_cycles: int = 6,
        h_cycles: int = 2,
        hidden_size: int = 2048,  # Default for Qwen-like models
        llm_hidden_size: int = 2048,
        expansion: float = 2.67,
        l_layers: int = 2,
        num_heads: int = 8,
        update_mode: str = 'lora', # or add_dora
        transcoder_layers = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.expansion = expansion
        self.l_layers = l_layers
        self.l_cycles = l_cycles
        self.h_cycles = h_cycles 
        self.num_heads = num_heads
        self.update_mode = update_mode  # e.g., 'add_dora' or 'add_lora'
        self.transcoder_layers = transcoder_layers

class TRMLoraLayer(LoraLayer):
    """
    TRM LoRA layer that wraps a base layer and overrides forward for inline recursion and low-rank delta.
    Reuses L_net and TRMTranscoder from trm_adapter.py.
    """
    def __init__(
        self,
        base_layer: nn.Module,
        config: TRMConfig,
        **kwargs
    ):
        super().__init__(base_layer, r=config.r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, **kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_hidden_size = config.llm_hidden_size

        # Reuse L_net for recursion steps
        self.l_net = L_net(
            self.hidden_size,
            self.llm_hidden_size,
            config.l_layers,
            config.num_heads,
            config.expansion
        )

        # Reuse TRMTranscoder, enhanced for low-rank
        self.transcoder = TRMTranscoder(
            self.hidden_size,
            self.llm_hidden_size,
            expansion=config.expansion,
            trm_transcoder_layers=config.trm_transcoder_layers  # Default; can be configurable
        )
        self.transcoder.final_proj = nn.Identity()  # Low-rank handled in forward

        # Initial states non learnable
        self.zH_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.zL_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)


        self.cycles = config.cycles
        self.update_mode = config.update_mode

    def hrm(self, zL: Float[Tensor, 'b h'], zH: Float[Tensor, 'b h'], context_hs: Float[Tensor, 'b h']) -> tuple[Float[Tensor, 'b h'], Float[Tensor, 'b h']]:
        """
        Hierarchical Recursion Module (HRM) adapted from trm_adapter.py.
        Runs fixed cycles with L_net.
        """
        zLs = zL.unsqueeze(1)  # [b 1 h]
        zHs = zH.unsqueeze(1)   # [b 1 h]

        # H_cycles-1 without grad
        with torch.no_grad():
            for H_step in range(self.h_cycles-1):
                for L_step in range(self.l_cycles):
                    zLs = self.l_net(zLs, zHs + context_hs)
                zHs = self.l_net(zHs, zLs)

        # 1 with grad
        for L_step in range(self.l_cycles):
            zLs = self.l_net(zLs, zHs + context_hs)
        zHs = self.l_net(zHs, zLs)

        for _ in range(self.cycles):
            # FIXME wrong, look at trm_adapter
            zLs = self.l_net(zLs, zHs + context_hs.unsqueeze(1))
            zHs = self.l_net(zHs, zLs)
        return zHs.squeeze(1), zLs.squeeze(1)

    def forward(
        self,
        hidden_states: Float[Tensor, 'b s h'],
        zH: Optional[Float[Tensor, 'b h']] = None,
        zL: Optional[Float[Tensor, 'b h']] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **kwargs
    ) -> Float[Tensor, 'b s h']:
        if self.disable_adapters:
            if self.active_adapter in self.lora_A.keys():
                self.lora_A[self.active_adapter].zero_grad()
                self.lora_B[self.active_adapter].zero_grad()
            return self.base_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)

        # Run base layer
        base_hidden = self.base_layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)  # [b s h]

        b, s, h = base_hidden.shape

        # Initialize zH and zL if not provided
        if zH is None:
            zH = self.zH_init.unsqueeze(0).expand(b, -1)
        if zL is None:
            zL = self.zL_init.unsqueeze(0).expand(b, -1)

        # Pool context hidden states (mean pool, ref context_hs usage)
        context_hs = base_hidden[:, -1, :]

        # Run HRM recursion
        zL_next, zH_next = self.hrm(zL, zH, context_hs)

        # Transcoder MLP to produce features (ref TRMTranscoder)
        features = self.transcoder(zH_next)

        # Low-rank projection with DoRA-style decomposition for stable addition
        if self.update_mode == 'add_dora':
            # Compute LoRA weight: B @ A (note: typically A is input, B output)
            # Assuming standard LoRA dims: lora_A [r, in_features], lora_B [out_features, r]
            # For simplicity, assume square hidden for delta
            lora_weight = self.lora_B.weight @ self.lora_A.weight  # [h, r] @ [r, h_exp?] Adjust dims as needed
            # Placeholder: assume features [b, h_exp], lora_weight [h_llm, h_exp]
            lora_weight = lora_weight.to(features.device)
            magnitude = lora_weight.norm(dim=1, keepdim=True)  # [h_llm, 1]
            direction = lora_weight / (magnitude + 1e-6)
            delta_dir = (direction @ features.T).T  # [b, h_llm]
            delta_norm = delta_dir.norm(dim=1, keepdim=True)  # [b, 1]
            mag_norm_scale = magnitude / (delta_norm + 1e-6)  # [h_llm, 1] / [b,1] broadcast
            delta = (delta_dir * mag_norm_scale.T) * (self.lora_alpha / self.r)
        else:  # Plain LoRA
            # Standard LoRA delta (using PEFT's method)
            delta = self.lora_A(features)  # [b, r] if features [b, in]
            delta = self.lora_B(delta)  # [b, out]
            delta = delta * (self.lora_alpha / self.r)

        # Apply delta to last token hidden (latent position assumption)
        modified_hidden = base_hidden * 1.0
        modified_hidden[:, -1, :] += delta  # Additive update; configurable for other ops

        return modified_hidden

# FIXME this is not how  a peft model works, see https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/model.py#L36
class TRMModel(LoraModel):
    """
    TRM LoRA Model that uses TRMLoraLayer as the tuner layer class.
    """
    def __init__(self, model, config: TRMConfig, **kwargs):
        super().__init__(model, config, **kwargs)
        self.config = config
        self.tuner_layer_cls = TRMLoraLayer

    def print_trainable_parameters(self) -> None:
        super().print_trainable_parameters()
        print(f"TRM-specific params: cycles={self.config.cycles}, expansion={self.config.expansion}")

# Helper function to load the model with TRMConfig (can be imported/used in load_model.py)
def get_trm_model(base_model, config: TRMConfig):
    """
    Convenience function to load a PEFT model with TRM adapter.
    """
    return get_peft_model(base_model, config, adapter_name="default")

# For backward compatibility or direct instantiation
def create_trm_model(base_model, config: TRMConfig):
    return TRMModel(base_model, config)
