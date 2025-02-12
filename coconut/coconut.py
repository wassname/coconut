# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple

from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from typing import Tuple, List, Union, Optional, Dict
from torch import Tensor

from transformers import DynamicCache
# from transformers.models.gpt2 import GPT2LMHeadModel

from loguru import logger


from transformers import (
    Qwen2ForCausalLM,
    DynamicCache,
    PreTrainedTokenizer,
    Qwen2Config,
)


Outputs = namedtuple(
    "Outputs", ["loss", "inputs_embeds", "logits", "past_key_values", "hidden_states", "log",] # loss_ar loss_vcr
)
MAX_N_LATENT = 8

HiddenState = Float[Tensor, "b t h"]
HiddenStates = Tuple[Float[Tensor, "b t h"]]


# def batch_cov(x: Float[Tensor, 'b h']) -> Float[Tensor, 'b h h']:
#     """
#     apply cov only on last dim
#     """
#     N = x.shape[1]
#     xc = x - x.mean(dim=1, keepdim=True)
#     prods = torch.bmm(xc.unsqueeze(2), xc.unsqueeze(1))
#     return prods / (N - 1)  # Unbiased estimate


def calc_distribution_loss(
    hs: Float[Tensor, "b t h"], chunk_size: int = 32
) -> Float[Tensor, ""]:
    B, T, P = hs.shape
    total_loss = 0.0
    eps = 1e-6

    # hs = F.layer_norm(hs)

    x = rearrange(hs, "b t h -> t b h")

    # Get distribution per timestep
    x_dist = F.softmax(x, dim=-1)

    # Compute uniform target
    uniform = torch.ones_like(x_dist) / P

    x_dist_log = torch.log(x_dist + eps)

    # KL divergence from uniform (prevent collapse)
    kl_loss = F.kl_div(x_dist_log, uniform, reduction="batchmean", log_target=False)

    # Entropy term (maximize feature usage)
    entropy = -(x_dist * x_dist_log).sum(-1).mean()

    total_loss = kl_loss - 0.1 * entropy

    return total_loss / T


def calc_seq_vcr_loss(hs: Float[Tensor, "b t h"], η = 1e-4) -> Float[Tensor, ""]:
    B, T, P = hs.shape

    # Compute covariance per timestep across batch
    x = rearrange(hs, "b t h -> t b h")  # Shape: (T, B, P)

    # Calculate mean and center per timestepx.dtype
    x_mean = x.mean(dim=1, keepdim=True).detach()  # Shape: (T, 1, P)
    x_centered = x - x_mean

    # Compute covariance matrices for each timestep
    C = torch.bmm(x_centered.transpose(1, 2), x_centered) / (B - 1)  # Shape: (T, P, P)

    # Setup mask for diagonal elements
    diag = torch.eye(P, dtype=torch.bool, device=x.device).detach()

    # Calculate losses
    # TODO move these to function arguments. And try to make the loss automatically balanced if I can
    λ1, λ2 = 1. / 5000, 1. / 50000
    
    # The Variance Term encourages unit variance in each dimension (most important)
    var_loss = torch.relu(1 - torch.sqrt(C * diag + η))
    var_loss = reduce(var_loss, "t h1 h2 -> ", "sum") / (T * diag.sum())

    # the Covariance Term penalizes covariance between different dimensions, promoting decorrelation and diversity in representations
    non_diag = (~diag).detach()
    cov_loss = (C * non_diag).pow(2)
    cov_loss = reduce(cov_loss, "t h1 h2 -> ", "sum") / (T * non_diag.sum())

    # Combine and reduce
    loss = λ1 * var_loss + λ2 * cov_loss
    return loss, {"loss_vcr_var": var_loss.item(), "loss_vcr_cov": cov_loss.item()}




class VCRLoss(nn.Module):
    def __init__(self, H=1536, D=256):
        # TODO set these param properly
        super().__init__()
        self.down_proj = nn.Linear(
            H, D, bias=False
        )  # FIXME is this meant to be learnable?
        nn.init.orthogonal_(self.down_proj.weight)
        self.down_proj.weight.requires_grad = False
        # FIXME we should be applying this to the final high dim output too, but wait for reference implementation https://github.com/rarefin/SEQ_VCR

    def forward(self, hs_l: Float[Tensor, "l b t h"]) -> Float[Tensor, "b"]:
        # for each layer
        loss = 0
        logs = defaultdict(int)
        for hs in hs_l:
            hs = self.down_proj(hs)
            # loss += calc_distribution_loss(hs)
            loss_i, extra = calc_seq_vcr_loss(hs)
            for k, v in extra.items():
                logs[k] += v
            loss += loss_i
            torch.cuda.empty_cache()

        return loss, dict(logs)


def get_supressed_activations(
    hs: Float[Tensor, "l b t h"], w_out=None, w_inv=None
) -> Float[Tensor, "l b t h"]:
    """
    Novel experiment: Here we define a transform to isolate supressed activations, where we hypothesis that style/concepts/scratchpads and other internal only representations must be stored.

    See the following references for more information:

    - https://arxiv.org/pdf/2401.12181
        - > Suppression neurons that are similar, except decrease the probability of a group of related tokens

    - https://arxiv.org/html/2406.19384
        - > Previous work suggests that networks contain ensembles of “prediction" neurons, which act as probability promoters [66, 24, 32] and work in tandem with suppression neurons (Section 5.4).

    - https://arxiv.org/pdf/2401.12181
        > We find a striking pattern which is remarkably consistent across the different seeds: after about the halfway point in the model, prediction neurons become increasingly prevalent until the very end of the network where there is a sudden shift towards a much larger number of suppression neurons.
    """
    with torch.no_grad():
        # here we pass the hs through the last layer, take a diff, and then project it back to find which activation changes lead to supressed
        hs2 = rearrange(hs[:, :, -1:], "l b t h -> (l b t) h")
        hs_out2 = torch.nn.functional.linear(hs2, w_out)
        hs_out = rearrange(
            hs_out2, "(l b t) h -> l b t h", l=hs.shape[0], b=hs.shape[1], t=1
        )
        diffs = hs_out[:, :, :].diff(dim=0)
        diffs2 = rearrange(diffs, "l b t h -> (l b t) h")
        # W_inv = get_cache_inv(w_out)
        diffs_inv2 = torch.nn.functional.linear(diffs2, w_inv)
        diffs_inv = rearrange(
            diffs_inv2, "(l b t) h -> l b t h", l=hs.shape[0] - 1, b=hs.shape[1], t=1
        ).to(w_out.dtype)
        # TODO just return this?
        eps = 1.0e-1
        supressed_mask = (diffs_inv < -eps).to(hs.dtype)
        # supressed_mask = repeat(supressed_mask, 'l b 1 h -> l b t h', t=hs.shape[2])
    supressed_act = hs[1:] * supressed_mask
    return supressed_act


def hs2ie(
    hidden_states: HiddenStates,
    inputs_embeds: HiddenState,
    w_out=None,
    w_inv=None,
    method="-1",
) -> HiddenState:
    """hidden states to inputs_embeds"""

    n = len(hidden_states)
    i_half = int(n * 0.5)
    if method == "-1":
        return hidden_states[-1]
    elif method == "-2":
        return hidden_states[-2]
    elif method == "0.5":
        return hidden_states[i_half]

    # FIXME ok so this doesn't account for information being removed then added
    # and it assumed removal == reduction in positive magnitude, but it could be negative. So I should refactor for all reduction in magnitude
    hs = rearrange(list(hidden_states), "l b t h -> l b t h")
    supressed_act = get_supressed_activations(hs, w_out, w_inv)

    if method == "ie+supressed[-1]":
        # need to make it more like the original hidden states, so prev input embedding plus the supressed tokens
        return inputs_embeds + supressed_act[-1]  # last layer, add dummy sequence dim
    elif method == "ie+supressed[0.5:]":
        # ie comes in with the wrong shape and looking ahead
        return inputs_embeds + supressed_act[i_half:].sum(dim=0)
    elif method == "hs+supressed[0.5:]":
        return hidden_states[-1] + supressed_act[i_half:].sum(dim=0)
    # elif method == 'supressed[0.5:]':
    #     # FIXME this need to be repeated along token dim
    #     T = inputs_embeds.shape[1]
    #     hs = supressed_act[i_half:].sum(dim=0)
    #     return inputs_embeds + supressed_act[-1]
    elif method == "ie+supressed[0.5:]":
        return inputs_embeds + supressed_act[i_half:].sum(dim=0)
    elif method == "hs+supressed[0.5:]":
        return hidden_states[-1] + supressed_act[i_half:].sum(dim=0)
    elif method == "supressed[0.5:]":
        T = inputs_embeds.shape[1]
        hs = supressed_act[i_half:].sum(dim=0)
        # hs = repeat(hs, 'b 1 h -> b t h', t=T)
        return hs

    raise ValueError(f"Unknown method {method}")


class CoconutConfig(Qwen2Config):
    def __init__(self, **kwargs):
        self.replacement_method = kwargs.pop("replacement_method", "-1")
        self.latent_token_id = kwargs.pop("latent_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        super().__init__(**kwargs)


class CoconutQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.gen_forward_cnt = 0
        self.vcr_loss = VCRLoss()

    def forward(
        self, input_ids, attention_mask=None, labels=None, position_ids=None, **kwargs
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        if labels is None:
            labels = input_ids.clone()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

        logits = []

        latent_indices = (
            input_ids == self.config.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        ncr = (0, input_ids.shape[1])
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if max_n_latents > 0:
            ncr = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        all_hs = []

        # FIXME: this lags behind, but for efficiency we accept this limitation
        Wo = self.get_output_embeddings().weight
        Wo_inv = torch.pinverse(Wo.clone().float()).detach()
        device_type = input_ids.device.type

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # first forward pass
                with torch.autocast(device_type=device_type):
                    outputs = super().forward(
                        inputs_embeds=inputs_embeds[:, ncr[0] : ncr[1], :],
                        attention_mask=attention_mask[:, ncr[0] : ncr[1]],
                        position_ids=position_ids[:, ncr[0] : ncr[1]],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : ncr[0], :],
                        v[:, :, : ncr[0], :],
                    )
                    for k, v in kv_cache
                ]

                # Qwen needs this
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                with torch.autocast(device_type=device_type):
                    outputs = super().forward(
                        inputs_embeds=inputs_embeds[:, ncr[0] : ncr[1], :],
                        attention_mask=attention_mask[:, : ncr[1]],
                        position_ids=position_ids[:, ncr[0] : ncr[1]],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                    )

                hidden_states_offset = ncr[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            ncr = (
                ncr[1],
                (input_ids.shape[1] if pass_idx + 1 >= max_n_latents else ncr[1] + 1),
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
                recrv_embeds = hs2ie(
                    hidden_states,
                    inputs_embeds[:, :token_idx],
                    Wo,
                    Wo_inv,
                    method=self.config.replacement_method,
                )
                # print({'hs': torch.stack(hidden_states).shape, 'recrv_embeds': recrv_embeds.shape, 'tensor_list': tensor_list[batch_idx][token_idx].shape})
                tensor_list[batch_idx][token_idx] = recrv_embeds[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                # print(tensor_list[batch_idx][token_idx].shape, recrv_embeds.shape, batch_idx, token_idx, token_idx - 1 - hidden_states_offset)

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

            # Collect hs
            hs = rearrange(list(hidden_states), "l b t h -> l b t h")
            all_hs.append(hs)

        past_key_values = (
            [
                (
                    k[:, :, : ncr[0], :],
                    v[:, :, : ncr[0], :],
                )
                for k, v in kv_cache
            ]
            if kv_cache
            else None
        )
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # final pass
        with torch.autocast(device_type=device_type):
            outputs = super().forward(
                inputs_embeds=inputs_embeds[:, ncr[0] : ncr[1], :],
                attention_mask=attention_mask[:, : ncr[1]],
                position_ids=position_ids[:, ncr[0] : ncr[1]],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )

        logits.append(outputs.logits)

        # Collect hs
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
        with torch.autocast(device_type=input_ids.device.type):
            loss_vcr, extra = self.vcr_loss(all_hs)
        # TODO report diff losses to wandb
        extra['loss_ar'] = loss.item()
        loss += loss_vcr

        assert torch.isfinite(loss).all(), f"Loss is {loss}"

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=list(all_hs),
            log=extra,
        )

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=32,
        output_embedding=False,
        **kwargs,
    ):
        self.gen_forward_cnt = 0

        # assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        lyr_embed = self.get_input_embeddings()

        tokens = input_ids.detach()

        # reuse the forward pass from training to go through all the inputs before gen this include latent thoughts
        outputs = self.forward(input_ids)
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = outputs.logits[:, -1].argmax(-1).detach().unsqueeze(1)
        tokens = torch.cat((tokens, next_token), dim=1)
        new_token_embed = lyr_embed(next_token)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        kv_cache = outputs.past_key_values
        for _ in range(max_new_tokens - 1):
            # FIXME here we use the base model forward, that means we DO NOT use latent thoughts after the preconfigured ones
            outputs = super().forward(
                inputs_embeds=new_inputs_embeds, past_key_values=kv_cache
            )
            kv_cache = outputs.past_key_values
            self.gen_forward_cnt += 1
            next_token = outputs.logits[:, -1].argmax(-1).detach().unsqueeze(1)
            if (next_token == self.config.latent_token_id).all():
                logger.error("Latent token generated, not implemented in gen")

            tokens = torch.cat((tokens, next_token), dim=1)
            if (tokens == self.config.eos_token_id).any(1).all(0):
                break
            new_token_embed = lyr_embed(next_token)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if output_embedding:
            # for analysis purpose
            return tokens, new_inputs_embeds

        else:
            return tokens
