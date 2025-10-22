# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from collections import defaultdict
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from typing import Tuple, List, Union, Optional, Dict
from torch import Tensor

from transformers import DynamicCache
# from transformers.models.gpt2 import GPT2LMHeadModel

from loguru import logger


from transformers import (
    DynamicCache,
    PreTrainedTokenizer,
    Qwen2Config, Qwen3Config,
    LlamaConfig,
    LlamaForCausalLM, PreTrainedModel
)

from coconut.hs2ie import hs2ie, get_supressed_activations
from coconut.configs import BaseConfig


Outputs = namedtuple(
    "Outputs", ["loss", "inputs_embeds", "logits", "past_key_values", "hidden_states", "log", "input_embed_diff" ] # loss_ar loss_vcr
)

# max during gen
MAX_N_LATENT = 8

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

    loss_fct = CrossEntropyLoss(reduction='none')
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(-1, shift_logits.size(1)) # [b, s]
    loss_per_token = loss_per_token * shift_mask.float()
    nll = (loss_per_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    return nll, loss_per_token

class Coconut(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: BaseConfig
    ):
        super().__init__()
        self.model = base_model

        self.config = config

        self.gen_forward_cnt = 0

        # TRM LoRA mode: freeze base LLM parameters (PEFT handles adapters)
        if getattr(self.config, 'use_trm_lora', False):
            logger.info("Freezing base LLM parameters for TRM LoRA")
            for param in self.model.base_model.parameters():
                param.requires_grad = False
            self.model.enable_input_require_grads()


    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None, collect_hs=False, **kwargs):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        all_labels = input_ids.clone()
        # remove latents from loss computation
        all_labels[input_ids == self.config.latent_token_id] = -100
        all_labels[input_ids == self.config.bot_token_id] = -100
        all_labels[input_ids == self.config.eot_token_id] = -100

        # Compute base NLL without adapter if margin loss enabled
        nll_base = None
        if self.config.loss_nll_ratio_margin and all_labels is not None:
            if hasattr(self.model, 'disable_adapters'):
                self.model.disable_adapters()
            with torch.no_grad():
                base_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=False,
                )
                nll_base = get_nll(base_outputs.logits, all_labels, attention_mask)[1]
                nll_base = nll_base.detach()
            self.model.enable_adapters()  # Re-enable if possible
            del base_outputs

        # Single forward pass with inline adapter
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=collect_hs,
            **kwargs
        )

        logits = outputs.logits
        all_hs = outputs.hidden_states if collect_hs else None

        self.gen_forward_cnt += 1

        # Losses
        question_nll = get_nll(logits, all_labels, attention_mask)[1]

        loss_diff = 0.0
        # No input_embed_diff with inline adapter; set to 0 or skip reg

        answer_loss = get_nll(logits, labels, attention_mask)[0] if labels is not None else torch.tensor(0.0)

        question_margin_loss = torch.tensor(0.0)
        if self.config.loss_nll_ratio_margin and nll_base is not None:
            question_margin_loss = torch.mean(F.relu(question_nll - nll_base - 0.1) ** 4)

        total_loss = answer_loss + question_margin_loss + loss_diff

        extra = {
            "loss/answer": answer_loss,
            "loss/question_margin": question_margin_loss,
            'loss/input_embed_diff': loss_diff,
            "loss/total": total_loss,
            'nll/question': question_nll.mean().item(),
        }
        if nll_base is not None:
            extra['nll/base'] = nll_base.mean().item()
        extra = {k: v.mean().detach().cpu().item() if isinstance(v, torch.Tensor) else v for k, v in extra.items()}

        assert torch.isfinite(total_loss).all(), f"Loss is {total_loss}"

        return Outputs(
            loss=total_loss,
            inputs_embeds=None,  # Not used now
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=all_hs,
            input_embed_diff=None,
            log=extra
        )


    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        min_new_tokens=1,
        output_embedding=False,
        **kwargs,
    ):
        """Generate tokens using the base model.generate, with inline adapter active."""
        self.gen_forward_cnt = 0

        # Use model.generate for efficient KV caching and inline adapter
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,  # Greedy for consistency
                pad_token_id=self.config.eos_token_id,
                eos_token_id=self.config.eos_token_id,
                **kwargs
            )

        if output_embedding:
            # For analysis, return tokens and last embeds (simplified)
            last_embeds = self.model.get_input_embeddings()(generated_tokens[:, input_ids.shape[1]:])
            return generated_tokens, last_embeds
        else:
            return generated_tokens

def check_input_lens(input_ids, attention_mask, kv_cache):
    # Handle None cache
    if kv_cache is None:
        len_c = 0
    # Handle DynamicCache - use get_seq_length() method
    elif hasattr(kv_cache, 'get_seq_length'):
        len_c = kv_cache.get_seq_length()
    else:
        # legacy cache format
        len_c = kv_cache.key_cache[0].shape[2] if len(kv_cache.key_cache) else 0
    
    len_ids = input_ids.shape[1]
    len_att = attention_mask.shape[1]
    assert len_c + len_ids == len_att, f"The length of the attention mask  {len_att} should cover the cache length {len_c} and the len of the input_ids {len_ids}"
