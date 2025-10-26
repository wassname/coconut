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

from contextlib import contextmanager
# from coconut.hs2ie import hs2ie, get_supressed_activations
from coconut.configs import BaseConfig
from coconut.adapters import set_adapter
from coconut.trmlora.recursive_lora import TRMLoraLayer
from coconut.trmlora.recursive_delora import TRMDeloraLayer
from coconut.trmlora.recursive_hra import TRMHraLayer


Outputs = namedtuple(
    "Outputs",
    [
        "loss",
        "inputs_embeds",
        "logits",
        "past_key_values",
        "hidden_states",
        "log",
        # "input_embed_diff"
    ],  # loss_ar loss_vcr
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

    loss_fct = CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(-1, shift_logits.size(1))  # [b, s]
    loss_per_token = loss_per_token * shift_mask.float()
    nll = (loss_per_token * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    return nll, loss_per_token


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
    def recursion_context(self, cache: dict, steering_mode=None):
        """Context manager to inject recursion_cache into all TRM layers."""
        # Walk down model tree and inject cache into TRM layers
        trm_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (TRMLoraLayer, TRMDeloraLayer, TRMHraLayer)):
                trm_layers.append(module)
                if name not in cache:
                    cache[name] = {}
                if steering_mode is not None:
                    cache[name]['steering_mode'] = steering_mode
                module._recursion_cache = cache[name]
        
        try:
            yield cache
        finally:
            # Clean up: remove cache from all TRM layers
            for module in trm_layers:
                module._recursion_cache = None

    def forward(
        self, input_ids, attention_mask=None, labels=None, position_ids=None, **kwargs
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        if labels is None:
            labels = input_ids.clone()
        if position_ids is None:
            position_ids = (
                torch.arange(
                    0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
                )
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )

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

        logits = []

        latent_indices = (
            input_ids == self.config.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        a, b = 0, input_ids.shape[1]
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if max_n_latents > 0:
            a,b = 0, latent_indices[:, 1].min().item()
            # before the earliest latent token position

        kv_cache = None

        all_hs = []

        recursion_cache = {}

        # Three-stage processing: (1) before latents, (2) latents one-by-one, (3) after latents
        # Each stage is a forward pass through the model on a different portion of the sequence
        
        # STAGE 1: Before first latent (base model only)
        if max_n_latents > 0:
            with torch.no_grad():
                with set_adapter(self.model, None):
                    outputs = self.model.forward(
                        inputs_embeds=inputs_embeds[:, a:b],
                        attention_mask=attention_mask[:, :b],
                        position_ids=position_ids[:, a:b],
                        past_key_values=kv_cache,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    logits.append(outputs.logits)
                    kv_cache = outputs.past_key_values
                    
                    if self.config.collect_hs:
                        hs = rearrange(
                            list(outputs.hidden_states),
                            "l b t h -> l b t h",
                        ).detach().cpu()
                        all_hs.append(hs)
                    
                    a = b  # Move to first latent position
            
            # STAGE 2: Latent tokens - one forward pass per token (adapter + recursion active)
            with self.recursion_context(recursion_cache):
                for pass_idx in range(max_n_latents):

                    # Process one latent token
                    b = a + 1
                    
                    with set_adapter(self.model, self.model.active_adapter):
                        outputs = self.model.forward(
                            inputs_embeds=inputs_embeds[:, a:b],
                            attention_mask=attention_mask[:, :b],
                            position_ids=position_ids[:, a:b],
                            recursion_cache=recursion_cache,
                            past_key_values=kv_cache,  # Cache already has exactly positions [0:a]
                            output_hidden_states=True,
                            use_cache=True,
                        )

                        logits.append(outputs.logits)

                        kv_cache = outputs.past_key_values
                        assert kv_cache is not None

                        if self.config.collect_hs:
                            hs = rearrange(
                                list(outputs.hidden_states),
                                "l b t h -> l b t h",
                            ).detach().cpu()
                            all_hs.append(hs)
                        
                        a = b  # Move to next position
                
                assert len(recursion_cache) > 0, "Recursion cache should be populated after latent processing"

        # STAGE 3: After latents (base model only)
        # If no latents exist (max_n_latents==0), a=0 so this processes entire sequence
        b = input_ids.shape[1]
        if a < b:  # True when: (1) tokens remain after latents, or (2) no latents at all

            has_zH = len(recursion_cache) and ('zH' in next(iter(recursion_cache.values())))  # just to make sure it's not empty

            # in peristent steering mode, the learned latent is applied even after the <latent> tokens
            if self.config.trm_persistent_steering and has_zH:
                with self.recursion_context(recursion_cache, steering_mode=True):
                    with set_adapter(self.model, self.model.active_adapter):
                        outputs = self.model.forward(
                            inputs_embeds=inputs_embeds[:, a:b],
                            attention_mask=attention_mask[:, :b],
                            position_ids=position_ids[:, a:b],
                            past_key_values=kv_cache,
                            output_hidden_states=True,
                            use_cache=True,
                        )
            else:
                with set_adapter(self.model, None):
                    outputs = self.model.forward(
                        inputs_embeds=inputs_embeds[:, a:b],
                        attention_mask=attention_mask[:, :b],
                        position_ids=position_ids[:, a:b],
                        past_key_values=kv_cache,
                        output_hidden_states=True,
                        use_cache=True,
                    )

            logits.append(outputs.logits)

            # collect hs
            if self.config.collect_hs:
                hs = rearrange(list(outputs.hidden_states), "l b t h -> l b t h").detach().cpu()
                all_hs.append(hs)
                all_hs = torch.concat(all_hs, dim=2)


        logits = torch.cat(logits, dim=-2)

        # Answer loss (primary objective)
        answer_loss, _ = get_nll(logits, labels, attention_mask)


        losses = {
            "answer_loss": answer_loss,
        }
        question_nll = torch.tensor(0.0, device=input_ids.device)
        if self.config.loss_nll_ratio_margin:
            question_nll, loss_per_token = get_nll(logits, all_labels, attention_mask)              
            # Question margin loss (regularization: penalize if NLL > threshold)
            question_margin_loss = torch.mean(F.relu(question_nll - nll_base - .1) ** 4)
            losses['question_margin_loss'] = question_margin_loss

        extra = {
            'nll/question': question_nll,
            'nll/base': nll_base,            
        }

        # Seq-VCR loss
        # in the paper they apply it to the last hidden state, we apply it to all. But it should be applied to the most relevent place, e.g. zH, cache. Or replace with topoloss which has more code support
        if self.config.loss_seq_vcr:
            with torch.autocast(device_type=input_ids.device.type):
                loss_vcr, extra2 = self.vcr_loss(all_hs)
            extra['loss_ar'] = loss_vcr.item()
            extra.update(extra2)
            losses['seq_vcr_loss'] = loss_vcr


        # Combined loss
        total_loss = sum(losses.values())
        assert torch.isfinite(total_loss).all(), f"Loss is {total_loss}"
        extra['loss/total'] = total_loss
        for k, v in losses.items():
            extra[f"loss/{k}"] = v

        extra = {k: v.mean().detach().cpu().item() for k, v in extra.items()}

        return Outputs(loss=total_loss, inputs_embeds=inputs_embeds, logits=logits, past_key_values=outputs.past_key_values,
                        hidden_states=list(all_hs), 
                        log=extra)

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        min_new_tokens=1,
        **kwargs,
    ):
        """Generate answer tokens after processing latent reasoning.

        Note: Requires input_ids to contain pre-filled <latent> tokens.
        The method processes these latents with TRM (if enabled), then generates
        the answer continuation. For dynamic latent generation, use a different approach.
        """
        # self.gen_forward_cnt = 0

        # assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        lyr_embed = self.model.get_input_embeddings()

        tokens = input_ids.detach()
        T = input_ids.shape[1]

        # reuse the forward pass from training to go through all the inputs before gen, this includes latent thoughts
        # Use self.forward (Coconut) not self.model.forward (base LLM) to enable TRM
        with torch.no_grad():
            coconut_outputs = self.forward(input_ids, attention_mask)
        outputs = type(
            "obj",
            (object,),
            {
                "logits": coconut_outputs.logits,
                "past_key_values": coconut_outputs.past_key_values,
            },
        )()

        # get the first token using the current hidden state
        next_token = outputs.logits[:, -1].argmax(-1).detach().unsqueeze(1)
        tokens = torch.cat((tokens, next_token), dim=1)
        new_inputs_embeds = lyr_embed(next_token)
        # new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
        B = tokens.shape[0]
        new_att_mask = torch.cat(
            (attention_mask, torch.ones((B, 1), device=attention_mask.device)), dim=1
        )

        # get other tokens
        kv_cache = outputs.past_key_values
        for _ in range(max_new_tokens - 1):
            # Use base model forward for answer generation (latents already processed above)
            check_input_lens(new_inputs_embeds, new_att_mask, kv_cache)
            outputs = self.model.forward(
                inputs_embeds=new_inputs_embeds,
                past_key_values=kv_cache,
                attention_mask=new_att_mask,
            )
            kv_cache = outputs.past_key_values
            # self.gen_forward_cnt += 1
            next_token = outputs.logits[:, -1].argmax(-1).detach().unsqueeze(1)
            if (next_token == self.config.latent_token_id).any():
                logger.error("Latent token generated, not implemented in gen")

            tokens = torch.cat((tokens, next_token), dim=1)

            # Allow it to stop early if all batch have generated EOS
            if (tokens[:, T:] == self.model.config.eos_token_id).any(1).all(0):
                if _ > min_new_tokens:
                    logger.info("EOS token generated, stopping early")
                    break

            new_inputs_embeds = lyr_embed(next_token)
            new_att_mask = torch.cat(
                (new_att_mask, torch.ones((B, 1), device=new_att_mask.device)), dim=1
            )
        return tokens


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
