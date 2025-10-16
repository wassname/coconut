# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
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
from coconut.vcr_loss import VCRLoss
from coconut.trm_adapter import CoconutTRM
from coconut.configs import BaseConfig


Outputs = namedtuple(
    "Outputs", ["loss", "inputs_embeds", "logits", "past_key_values", "hidden_states", "log", ] # loss_ar loss_vcr
)

# max during gen
MAX_N_LATENT = 8


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

        # FIXME this is getting quantised
        if self.config.loss_seq_vcr:
            self.vcr_loss = VCRLoss(H=self.config.hidden_size)
        
        # TRM mode: add TRM adapter
        if getattr(self.config, 'use_trm', False):
            logger.info("Initializing TRM adapter for frozen LLM")
            self.trm = CoconutTRM(
                hidden_size=self.model.config.hidden_size,
                trm_n_sup=self.config.trm_n_sup,
                n_detached_recursions=self.config.n_detached_recursions,
                num_layers=self.config.trm_num_layers,
                num_heads=self.config.trm_num_heads,
                expansion=self.config.trm_expansion,
                n_gradient_recursions=self.config.n_gradient_recursions
            )
            # Freeze LLM when using TRM
            logger.info("Freezing base LLM parameters")
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze the LM head to allow gradients from the loss
            for param in self.model.lm_head.parameters():
                param.requires_grad = True

            self.model.enable_input_require_grads()


    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None, **kwargs):

        
        if attention_mask is None:
            attention_mask=torch.ones_like(input_ids, device=input_ids.device)
        if labels is None:
            labels=input_ids.clone()
        if position_ids is None:
            position_ids=torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)


        logits = []

        latent_indices = (
            input_ids == self.config.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        all_hs = []

        for pass_idx in range(max_n_latents):
            # FIXME why does it have this when it also has TRM, we want TRM not this
            # TRM-style: detach gradients for early passes, keep gradients for last N passes
            should_detach = (
                self.training 
                and self.config.n_detached_recursions > 0 
                and pass_idx < (max_n_latents - self.config.n_detached_recursions)
            )
            
            if should_detach:
                ctx = torch.no_grad()
            else:
                ctx = torch.enable_grad()
            
            with ctx:
                if kv_cache is None:
                    # first forward pass
                    outputs = self.model.forward(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    hidden_states_offset = 0

                else:
                    # extract kv cache to reuse
                    past_key_values = [
                        (
                            k[:, :, : next_compute_range[0], :],
                            v[:, :, : next_compute_range[0], :],
                        )
                        for k, v in kv_cache
                    ]

                    # Qwen needs this
                    past_key_values= DynamicCache.from_legacy_cache(past_key_values)

                    outputs = self.model.forward(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[:, : next_compute_range[1]],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                    )

                    hidden_states_offset = next_compute_range[0]
                    # when we use kv_cache for the first k tokens
                    # in `outputs.hidden_states`, [0, k) will be skipped
                    # so we need to keep this offset to correctly use the last hidden states

                logits.append(outputs.logits)

                next_compute_range = (
                    next_compute_range[1],
                    (
                        input_ids.shape[1]
                        if pass_idx + 1 >= max_n_latents
                        else next_compute_range[1] + 1
                    ),
                )

                hidden_states = outputs.hidden_states
                assert hidden_states is not None
                kv_cache = outputs.past_key_values
                if isinstance(kv_cache, DynamicCache):
                    kv_cache = kv_cache.to_legacy_cache()
                assert kv_cache is not None

                # feedback the continuous thoughts to the input_embeds

                # first decide the positions to feedback
                filling_indices = [
                    (instance_idx, mask_list[pass_idx])
                    for instance_idx, mask_list in enumerate(latent_lists)
                    if len(mask_list) > pass_idx
                ]

                # to avoid in-place operations
                # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
                tensor_list = [
                    [
                        inputs_embeds[batch_idx, pos, :]
                        for pos in range(inputs_embeds.shape[1])
                    ]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]

                # replace some of them with continuous thoughts
                zL_prev, zH_prev = None, None
                for idx_pair in filling_indices:
                    batch_idx, token_idx = idx_pair

                    # FIXME should I not run .trm here instead
                    tensor_list[batch_idx][token_idx] = self._forward_trm(
                        input_ids,

                    )

                    latent_embed_single, zL_prev, zH_prev = self.trm(hidden_states, zL_prev, zH_prev, max_loops=max_loops)
                    tensor_list[batch_idx][token_idx] = latent_embed_single

                # assemble the new inputs_embeds
                inputs_embeds = torch.stack(
                    [
                        torch.stack(tensor_list[batch_idx])
                        for batch_idx in range(inputs_embeds.shape[0])
                    ]
                )
            
            # Detach inputs_embeds after detached passes to prevent gradients flowing back
            if should_detach:
                inputs_embeds = inputs_embeds.detach()

        past_key_values=(
                        [
                            (
                                k[:, :, : next_compute_range[0], :],
                                v[:, :, : next_compute_range[0], :],
                            )
                            for k, v in kv_cache
                        ]
                        if kv_cache
                        else None
                    )
        past_key_values= DynamicCache.from_legacy_cache(past_key_values)

        # final pass
        outputs = self.model.forward(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        # collect hs
        hs = rearrange(list(outputs.hidden_states), "l b t h -> l b t h")
        all_hs.append(hs)
        all_hs = torch.concat(all_hs, dim=2)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Seq-VCR loss
        # in the paper they apply it to the last hidden state, we apply it to all
        extra = {}
        if self.config.loss_seq_vcr:
            with torch.autocast(device_type=input_ids.device.type):
                loss_vcr, extra2 = self.vcr_loss(all_hs)
            extra['loss_ar'] = loss.item()
            extra.update(extra2)
            loss += loss_vcr

        assert torch.isfinite(loss).all(), f"Loss is {loss}"

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits, past_key_values=outputs.past_key_values,
                        hidden_states=list(all_hs), log=extra)


    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        min_new_tokens=1,
        output_embedding=False,
        **kwargs,
    ):
        # TODO: The `generate` method does not currently support TRM-style recursive reasoning.
        # It falls back to the base model's generation, which will not produce latent thoughts.
        # A full implementation would require iteratively calling the TRM during generation.
        self.gen_forward_cnt = 0

        # assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        lyr_embed = self.model.get_input_embeddings()

        tokens = input_ids.detach()
        T = input_ids.shape[1]

        # reuse the forward pass from training to go through all the inputs before gen, this includes latent thoughts
        outputs = self.model.forward(input_ids, attention_mask)

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
            # FIXME should be generate?
            # here we use the base model forward, that means we DO NOT use latent thoughts after the preconfigured ones
            check_input_lens(new_inputs_embeds, new_att_mask, kv_cache)
            outputs = self.model.forward(inputs_embeds=new_inputs_embeds, past_key_values=kv_cache, attention_mask=new_att_mask)
            kv_cache = outputs.past_key_values
            self.gen_forward_cnt += 1
            next_token = outputs.logits[:, -1].argmax(-1).detach().unsqueeze(1)
            if (next_token == self.config.latent_token_id).any():
                logger.error("Latent token generated, not implemented in gen")

            tokens = torch.cat((tokens, next_token), dim=1)

            # Allow it to stop early if all batch have generated EOS
            if (tokens[:, T:] == self.model.config.eos_token_id).any(1).all(0):
                if _>min_new_tokens:
                    logger.info("EOS token generated, stopping early")
                    break

            new_inputs_embeds = lyr_embed(next_token)
            new_att_mask = torch.cat(
                (new_att_mask, torch.ones((B, 1), device=new_att_mask.device)), dim=1
            )

        if output_embedding:
            # for analysis purpose
            return tokens, new_inputs_embeds

        else:
            return tokens

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
