# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from collections import defaultdict
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from typing import Tuple, List, Union, Optional, Dict
from torch import Tensor
from coconut.vcr_loss import VCRLoss
from transformers import DynamicCache
# from transformers.models.gpt2 import GPT2LMHeadModel

from loguru import logger


from transformers import (
    DynamicCache,
    PreTrainedTokenizer,
    Qwen2Config,
    Qwen3Config,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)

from contextlib import contextmanager, nullcontext

# from coconut.hs2ie import hs2ie, get_supressed_activations
from coconut.configs import BaseConfig
from coconut.adapters import set_adapter
from coconut.trmlora.recursive_lora import TRMLoraLayer
from coconut.trmlora.recursive_delora import TRMDeloraLayer
from coconut.trmlora.recursive_hra import TRMHraLayer
from coconut.trmlora.recursive_svft import TRMSvftLayer


Outputs = namedtuple(
    "Outputs",
    [
        "loss",
        # "inputs_embeds",
        "logits",
        "past_key_values",
        "hidden_states",
        "log",
        "recursion_cache",
        # "input_embed_diff"
    ],  # loss_ar loss_vcr
)

# max during gen
MAX_N_LATENT = 8


def kv_cache_shape(kv_cache: DynamicCache) -> tuple:
    # [layers, batch, n_heads, seq_len, head_dim]
    return (len(kv_cache.layers),) + tuple(kv_cache.layers[0].values.shape)


def get_nll(logits, labels=None, attention_mask=None):
    if labels is None:
        return torch.tensor(0.0), torch.tensor(0.0)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if attention_mask is None:
        shift_mask = torch.ones_like(shift_labels)
    else:
        shift_mask = attention_mask[..., 1:].contiguous().clone()

    # also mask the -100 loss positions
    shift_mask[shift_labels == -100] = 0

    loss_fct = CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(-1, shift_logits.size(1))  # [b, s]
    loss_per_token = loss_per_token * shift_mask.float()
    nll = (loss_per_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    return nll, loss_per_token

@contextmanager
def recursion_context(model, input_ids: Tensor, cache: dict, latent_token_id):
    """Context manager to inject recursion_cache into all TRM layers."""
    # Walk down model tree and inject cache into TRM layers
    latent_mask = (input_ids == latent_token_id).detach().cpu()  # B x T
    trm_layers = []
    for name, module in model.named_modules():
        if isinstance(
            module, (TRMLoraLayer, TRMDeloraLayer, TRMHraLayer, TRMSvftLayer)
        ):
            trm_layers.append(module)
            if name not in cache:
                cache[name] = {}
            cache[name]["latent_mask"] = latent_mask
            module._recursion_cache = cache[name]

    try:
        yield cache
    finally:
        # Clean up: remove cache from all TRM layers
        for module in trm_layers:
            module._recursion_cache = None
            
class Coconut(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: BaseConfig):
        super().__init__()
        self.model = base_model

        self.config = config

        if self.config.loss_seq_vcr:
            self.vcr_loss = VCRLoss(H=self.config.lora_r)

        # self.gen_forward_cnt = 0
        self._recursion_cache = None  # Will be set via context manager

    @contextmanager
    def recursion_context(self, input_ids: Tensor, cache: dict):
        """Context manager to inject recursion_cache into all TRM layers."""
        with recursion_context(self.model, input_ids, cache, self.config.latent_token_id):
            yield cache

    @contextmanager
    def with_adapter_and_recursion(
        self,
        enable_adapter: Optional[Union[bool, str]] = True,
        recursion_cache: Optional[dict] = None,
        steering_mode: Optional[bool] = None,
    ):
        """Combined context manager: enables adapter and/or recursion as specified."""
        adapter_name = (
            self.model.active_adapter
            if enable_adapter is True
            else (enable_adapter if isinstance(enable_adapter, str) else None)
        )
        adapter_ctx = set_adapter(self.model, adapter_name)

        if recursion_cache is not None:
            recursion_ctx = self.recursion_context(
                recursion_cache, steering_mode=steering_mode
            )
        else:
            recursion_ctx = nullcontext()

        with adapter_ctx:
            with recursion_ctx:
                yield

    def forward(
        self, input_ids, attention_mask=None, labels=None, position_ids=None, **kwargs
    ):

        all_labels = input_ids.clone()
        # remove latents from loss computation
        all_labels[input_ids == self.config.latent_token_id] = -100
        all_labels[input_ids == self.config.bot_token_id] = -100
        all_labels[input_ids == self.config.eot_token_id] = -100

        # Compute base NLL without adapter if margin loss enabled
        nll_base = torch.tensor(0.0, device=input_ids.device)
        assert self.model.active_adapter is not None, (
            "Adapter must be active during forward"
        )
        if self.config.loss_nll_ratio_margin and all_labels is not None:
            # Here we just do one full run of the base model is we need it for the extra loss
            with set_adapter(self.model, None):
                with torch.no_grad():
                    base_outputs = self.model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_hidden_states=False,
                    )

                    nll_base = get_nll(base_outputs.logits, all_labels, attention_mask)[
                        1
                    ]
                    nll_base = nll_base.detach()
                    del base_outputs

        recursion_cache = {}
        with self.recursion_context(input_ids, cache=recursion_cache):
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=False,
            )

        # Answer loss (primary objective)
        answer_loss, _ = get_nll(outputs.logits, labels, attention_mask)

        losses = {
            "answer_loss": answer_loss,
        }
        question_nll = torch.tensor(0.0, device=input_ids.device)
        if self.config.loss_nll_ratio_margin:
            question_nll, loss_per_token = get_nll(
                outputs.logits, all_labels, attention_mask
            )
            # Question margin loss (regularization: penalize if NLL > threshold)
            question_margin_loss = torch.mean(
                F.relu(question_nll - nll_base - 0.1) ** 4
            )
            losses["question_margin_loss"] = question_margin_loss

        extra = {
            "nll/question": question_nll,
            "nll/base": nll_base,
        }

        # Combined loss
        total_loss = sum(losses.values())
        assert torch.isfinite(total_loss).all(), f"Loss is {total_loss}"
        extra["loss/total"] = total_loss
        for k, v in losses.items():
            extra[f"loss/{k}"] = v

        extra = {k: v.mean().detach().cpu().item() for k, v in extra.items()}

        recursion_cache = (
            {k: {kk: v.detach().cpu() for kk, v in v.items()} for k, v in recursion_cache.items()}
            if len(recursion_cache) > 0
            else None
        )

        return Outputs(
            loss=total_loss,
            recursion_cache=recursion_cache,
            log=extra,

            hidden_states=outputs.hidden_states,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )

    def generate(
        self,
        input_ids,
        attention_mask,
        **kwargs,
    ):
        """Generate answer tokens after processing latent reasoning.

        Note: Requires input_ids to contain pre-filled <latent> tokens.
        The method processes these latents with TRM (if enabled), then generates
        the answer continuation. For dynamic latent generation, use a different approach.
        """
        recursion_cache = {}
        with self.recursion_context(input_ids, cache=recursion_cache):
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,  **kwargs)

        return outputs


def check_input_lens(input_ids, attention_mask, kv_cache):
    # Handle None cache
    if kv_cache is None:
        len_c = 0
    # Handle DynamicCache - use get_seq_length() method
    elif hasattr(kv_cache, "get_seq_length"):
        len_c = kv_cache.get_seq_length()
    else:
        # legacy cache format
        len_c = kv_cache.key_cache[0].shape[2] if len(kv_cache.key_cache) else 0

    len_ids = input_ids.shape[1]
    len_att = attention_mask.shape[1]
    assert len_c + len_ids == len_att, (
        f"The length of the attention mask  {len_att} should cover the cache length {len_c} and the len of the input_ids {len_ids}"
    )
