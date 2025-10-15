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
    Qwen2ForCausalLM, Qwen3ForCausalLM,
    DynamicCache,
    PreTrainedTokenizer,
    Qwen2Config, Qwen3Config,
    LlamaConfig,
    LlamaForCausalLM,
)

from coconut.hs2ie import hs2ie, get_supressed_activations
from coconut.vcr_loss import VCRLoss
from coconut.trm_adapter import CoconutTRM


Outputs = namedtuple(
    "Outputs", ["loss", "inputs_embeds", "logits", "past_key_values", "hidden_states", "log", ] # loss_ar loss_vcr
)

# max during gen
MAX_N_LATENT = 8




class CoconutConfig(Qwen3Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # to set extra attributes from kwargs they need to be set
        self.replacement_method = None
        self.latent_token_id = None
        self.eos_token_id = None
        self.use_position_ids = None
        self.loss_seq_vcr = None
        self.n_detached_recursions = None
        self.use_trm = None
        self.load_in_4bit = None
        self.load_in_8bit = None
        self.trm_num_layers = None
        self.trm_num_heads = None
        self.trm_expansion = None

class CoconutQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(
        self,
        config: CoconutConfig
    ):
        super().__init__(config)
        assert self.config.latent_token_id is not None, "latent_token_id must be set in the config"
        assert self.config.eos_token_id is not None, "eos_token_id must be set in the config"
        assert self.config.use_position_ids is not None, "use_position_ids must be set in the config"
        assert self.config.loss_seq_vcr is not None, "loss_seq_vcr must be set in the config"
        assert self.config.replacement_method is not None, "replacement_method must be set in the config"
        assert self.config.n_detached_recursions is not None, "n_detached_recursions must be set in the config"

        self.gen_forward_cnt = 0

        # FIXME this is getting quantised
        if self.config.loss_seq_vcr:
            self.vcr_loss = VCRLoss(H=self.config.hidden_size)
        
        # TRM mode: add TRM adapter
        if getattr(self.config, 'use_trm', False):
            logger.info("Initializing TRM adapter for frozen LLM")
            self.trm = CoconutTRM(
                hidden_size=self.config.hidden_size,
                n_latents=4,  # Match typical coconut latent count
                n_detached=self.config.n_detached_recursions,
                num_layers=getattr(self.config, 'trm_num_layers', 2),
                num_heads=getattr(self.config, 'trm_num_heads', 8),
                expansion=getattr(self.config, 'trm_expansion', 2.67),
            )
            # Freeze LLM when using TRM
            logger.info("Freezing base LLM parameters")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False

            self.model.enable_input_require_grads()

    def _forward_trm(
        self, 
        input_ids: Int[Tensor, 'batch seq'], 
        attention_mask: Optional[Int[Tensor, 'batch seq']] = None,
        labels: Optional[Int[Tensor, 'batch seq']] = None,
        position_ids: Optional[Int[Tensor, 'batch seq']] = None,
        **kwargs
    ) -> Outputs:
        """
        TRM mode forward: frozen LLM + TRM adapter.
        
        Flow:
        1. Encode question with frozen LLM → context hidden states
        2. TRM adapter does recursive reasoning on latent states
        3. Decode with frozen LLM → logits
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        if labels is None:
            labels = input_ids.clone()
        
        # Find latent token positions
        latent_indices = (input_ids == self.config.latent_token_id).nonzero()
        logger.debug(f"TRM forward: found {len(latent_indices)} latent tokens")
        
        # Split into: question (before first latent) | latents | answer (after last latent)
        first_latent_pos = latent_indices[:, 1].min().item()
        last_latent_pos = latent_indices[:, 1].max().item()
        logger.debug(f"TRM forward: latent positions {first_latent_pos} to {last_latent_pos}")
        
        # Step 1: Encode question with frozen LLM
        question_ids: Int[Tensor, 'batch q_len'] = input_ids[:, :first_latent_pos]
        question_mask: Int[Tensor, 'batch q_len'] = attention_mask[:, :first_latent_pos]
        
        logger.debug(f"TRM forward: encoding question shape {question_ids.shape}")
        with torch.no_grad():
            question_outputs = self.model.forward(
                input_ids=question_ids,
                attention_mask=question_mask,
                output_hidden_states=True,
            )
        
        # Get context hidden states (last layer)
        context_hs: Float[Tensor, 'batch q_len hidden'] = question_outputs.hidden_states[-1]
        logger.debug(f"TRM forward: context_hs shape {context_hs.shape}")
        
        # Step 2: TRM recursive reasoning
        logger.debug("TRM forward: running TRM adapter")
        latent_embeds: Float[Tensor, 'batch n_latents hidden'] = self.trm(context_hs)
        logger.debug(f"TRM forward: latent_embeds shape {latent_embeds.shape}")
        
        # Step 3: Insert latent embeddings and decode
        # Build full input embeddings: question embeds | latent embeds | answer embeds
        embed_layer = self.get_input_embeddings()
        with torch.no_grad():
            question_embeds: Float[Tensor, 'batch q_len hidden'] = embed_layer(question_ids)
        
        # Get answer tokens (after latents)
        if last_latent_pos + 1 < input_ids.shape[1]:
            answer_ids: Int[Tensor, 'batch a_len'] = input_ids[:, last_latent_pos + 1:]
            with torch.no_grad():
                answer_embeds: Float[Tensor, 'batch a_len hidden'] = embed_layer(answer_ids)
            
            # Concatenate: question | latent_embeds | answer
            # latent_embeds has gradients, others don't
            full_embeds: Float[Tensor, 'batch full_len hidden'] = torch.cat([
                question_embeds, latent_embeds, answer_embeds
            ], dim=1)
        else:
            # No answer tokens, just question + latents
            full_embeds: Float[Tensor, 'batch full_len hidden'] = torch.cat([
                question_embeds, latent_embeds
            ], dim=1)
        
        # Decode with frozen LLM - but keep LM head WITH gradients for loss
        # Forward through transformer layers (frozen, no grad)
        with torch.no_grad():
            decode_outputs = super().model(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            final_hidden = decode_outputs[0]  # Last hidden state
        
        # Apply LM head WITH gradients (not frozen, allows gradients to flow to TRM)
        logits: Float[Tensor, 'batch seq vocab'] = self.lm_head(final_hidden)
        
        # Compute loss - now has gradient path through lm_head -> latent_embeds -> TRM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        
        extra = {'loss_ar': loss.item()}
        
        # Build hidden_states list (all layers from decode)
        hidden_states = decode_outputs.hidden_states if hasattr(decode_outputs, 'hidden_states') else (final_hidden,)
        
        return Outputs(
            loss=loss, 
            inputs_embeds=full_embeds, 
            logits=logits, 
            past_key_values=None,  # We don't use KV cache in TRM training
            hidden_states=list(hidden_states), 
            log=extra
        )

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None, **kwargs):
        # Route to TRM forward if enabled AND latents present
        if getattr(self.config, 'use_trm', False):
            latent_indices = (input_ids == self.config.latent_token_id).nonzero()
            if len(latent_indices) > 0:
                return self._forward_trm(input_ids, attention_mask, labels, position_ids, **kwargs)
            # If no latents, fall through to standard forward
        
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
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        all_hs = []

        # FIXME: this lags behind, but for efficiency we accept this limitation
        Wo = self.get_output_embeddings().weight
        # Wo_inv = torch.pinverse(Wo.clone().float()).detach()
        # device_type = input_ids.device.type

        for pass_idx in range(max_n_latents):
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
                    outputs = super().forward(
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

                    outputs = super().forward(
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
                # tensor_shape = torch.stack([torch.stack([xx for xx in x]) for x in tensor_list]).shape
                # # tensor_shapes = [[tuple(xx.shape) for xx in x] for x in tensor_list]
                # print({'pass_idx':pass_idx, 
                #        'inputs_embeds': inputs_embeds.shape, 
                #        'tensor_shape': tensor_shape, 
                #        'hidden_states_offset': hidden_states_offset})


                # replace some of them with continuous thoughts
                for idx_pair in filling_indices:
                    batch_idx, token_idx = idx_pair

                    # TODO experiment with transformers here, we are replacing. 
                    # replace it with the preceding last hidden states
                    Wo = self.get_output_embeddings().weight
                    recrv_embeds = hs2ie(hidden_states, inputs_embeds, Wo, method=self.config.replacement_method)
                    # print({'hs': torch.stack(hidden_states).shape, 'recrv_embeds': recrv_embeds.shape, 'tensor_list': tensor_list[batch_idx][token_idx].shape})
                    tensor_list[batch_idx][token_idx] = recrv_embeds[
                        batch_idx, token_idx - 1 - hidden_states_offset, :
                    ]
                    # print(tensor_list[batch_idx][token_idx].shape, recrv_embeds.shape, batch_idx, token_idx, token_idx - 1 - hidden_states_offset)


                # modification. Hypothesis: if the model has a unique positional id for thinking token then it will more quickly learn to mode switch to the recusrsive thinking mode
                if self.config.use_position_ids:
                    thinking_base_position = 100000  # Well beyond normal context windows
                    position_ids[batch_idx][token_idx] = thinking_base_position + pass_idx
                    # TODO consider token_type_ids, or add a distinct thinking vector the embeddings, perhaps just embedding <latent> token back in and adding


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
        outputs = super().forward(
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
        self.gen_forward_cnt = 0

        # assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        lyr_embed = self.get_input_embeddings()

        tokens = input_ids.detach()
        T = input_ids.shape[1]

        # reuse the forward pass from training to go through all the inputs before gen, this includes latent thoughts
        outputs = self.forward(input_ids, attention_mask)

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
            # here we use the base model forward, that means we DO NOT use latent thoughts after the preconfigured ones
            check_input_lens(new_inputs_embeds, new_att_mask, kv_cache)
            outputs = super().forward(inputs_embeds=new_inputs_embeds, past_key_values=kv_cache, attention_mask=new_att_mask)
            kv_cache = outputs.past_key_values
            self.gen_forward_cnt += 1
            next_token = outputs.logits[:, -1].argmax(-1).detach().unsqueeze(1)
            if (next_token == self.config.latent_token_id).any():
                logger.error("Latent token generated, not implemented in gen")

            tokens = torch.cat((tokens, next_token), dim=1)

            # Allow it to stop early if all batch have generated EOS
            if (tokens[:, T:] == self.config.eos_token_id).any(1).all(0):
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
    # Handle DynamicCache - use get_seq_length() method
    if hasattr(kv_cache, 'get_seq_length'):
        len_c = kv_cache.get_seq_length()
    else:
        # legacy cache format
        len_c = kv_cache.key_cache[0].shape[2] if len(kv_cache.key_cache) else 0
    
    len_ids = input_ids.shape[1]
    len_att = attention_mask.shape[1]
    assert len_c + len_ids == len_att, f"The length of the attention mask  {len_att} should cover the cache length {len_c} and the len of the input_ids {len_ids}"
