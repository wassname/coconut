from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from typing import Tuple, List, Union, Optional, Dict
from torch import Tensor
import torch
from collections import defaultdict
import torch.nn as nn

HiddenState = Float[Tensor, 'b t h']
HiddenStates = Tuple[Float[Tensor, 'b t h']]

global _w_out_inv
_w_out_inv = None

@torch.no_grad()
def get_cache_inv(w_out):

    # FIXME this will change each update, so we shouldn't cache it for more than a step
    global _w_out_inv

    if _w_out_inv is None:
        _w_out_inv = torch.pinverse(w_out.clone().float())
    return _w_out_inv

def get_supressed_activations(hs: Float[Tensor, 'l b t h'], w_out=None) -> Float[Tensor, 'l b t h']:
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
        hs2 = rearrange(hs[:, :, -1:], 'l b t h -> (l b t) h')
        hs_out2 = torch.nn.functional.linear(hs2, w_out)
        hs_out = rearrange(hs_out2, '(l b t) h -> l b t h', l=hs.shape[0], b=hs.shape[1], t=1)
        diffs = hs_out[:, :, :].diff(dim=0)
        diffs2 = rearrange(diffs, 'l b t h -> (l b t) h')
        W_inv = get_cache_inv(w_out)
        diffs_inv2 = torch.nn.functional.linear(diffs2, W_inv)
        diffs_inv = rearrange(diffs_inv2, '(l b t) h -> l b t h', l=hs.shape[0]-1, b=hs.shape[1], t=1).to(w_out.dtype)
        # TODO just return this?
        eps = 1.e-1
        supressed_mask = (diffs_inv < -eps).to(hs.dtype)
        # supressed_mask = repeat(supressed_mask, 'l b 1 h -> l b t h', t=hs.shape[2])
    supressed_act = hs[1:] * supressed_mask
    return supressed_act



def hs2ie(hidden_states: HiddenStates, inputs_embeds: HiddenState, w_out=None, method='-1') -> HiddenState:
    """hidden states to inputs_embeds"""

    n = len(hidden_states)
    i_half = int(n * 0.5)
    if method == '-1':
        return hidden_states[-1]
    elif method == '-2':
        return hidden_states[-2]
    elif method == '0.5':
        return hidden_states[i_half]
    


    # FIXME ok so this doesn't account for information being removed then added
    # and it assumed removal == reduction in positive magnitude, but it could be negative. So I should refactor for all reduction in magnitude
    hs = rearrange(list(hidden_states), 'l b t h -> l b t h')
    supressed_act = get_supressed_activations(hs, w_out)

    To = hidden_states[-1].shape[1]
    Ti = inputs_embeds.shape[1]

    if method == 'ie+supressed[-1]':
        # need to make it more like the original hidden states, so prev input embedding plus the supressed tokens
        return inputs_embeds[:, :To] + supressed_act[-1] # last layer, add dummy sequence dim
    elif method == 'ie+supressed[0.5:]':
        return inputs_embeds[:, :To] + supressed_act[i_half:].sum(dim=0)
    elif method == 'hs+supressed[0.5:]':
        return hidden_states[-1] + supressed_act[i_half:].sum(dim=0)
    elif method == 'supressed[0.5:]':
        # FIXME this need to be repeated along token dim
        hs = supressed_act[i_half:].sum(dim=0)
        return inputs_embeds[:, :To] + supressed_act[-1]
    elif method == 'ie+supressed[0.5:]':
        return inputs_embeds[:, :To] + supressed_act[i_half:].sum(dim=0)
    elif method == 'hs+supressed[0.5:]':
        return hidden_states[-1] + supressed_act[i_half:].sum(dim=0)
    elif method == 'supressed[0.5:]':
        hs = supressed_act[i_half:].sum(dim=0)
        hs = repeat(hs, 'b 1 h -> b t h', t=Ti)
        return hs
    
    ValueError(f"Unknown method {method}")
