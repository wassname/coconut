from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from typing import Tuple, List, Union, Optional, Dict
from torch import Tensor
import torch
from collections import defaultdict
import torch.nn as nn

HiddenState = Float[Tensor, 'b t h']
HiddenStates = Tuple[Float[Tensor, 'b t h']]

def _parse_frac_part(part: str, n: int) -> int:
    """
    turn “2” → 2, “-1” → -1, “0.5” → int(0.5 * n), “” → None
    """
    if part == "":
        return None
    if "." in part:
        # fractional index
        return int(float(part) * n)
    return int(part)

def lloc(spec: str, n_layers: int) -> Union[int, slice]:
    """
    spec examples:
      “-1”       → last layer
      “0.5”      → middle layer
      “2”        → layer 2
      “:”        → all layers
      “0.25:”    → from 25% to end
      “:0.75”    → from start to 75%
      “0.25:0.75”→ 25%–75% slice
    """
    if ":" in spec:
        start_str, end_str = spec.split(":", 1)
        a = _parse_frac_part(start_str, n_layers)
        b = _parse_frac_part(end_str,   n_layers)
        if a is None and b is None:
            return slice(None)
        elif a is None:
            return slice(None, b)
        elif b is None:
            return slice(a, None)
        else:
            return slice(a, b)
    else:
        a =_parse_frac_part(spec, n_layers)
        if a is None:
            return slice(None)
        else:
            return slice(a, a + 1)

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
    """
    hidden states to inputs_embeds

    We take in a method string which can be
    - hs[-1] (last layer)
    - hs[0.5:] (halfway through onwards)
    - hs[0.5] (halfway through)
    - supressed[-1:] (last layer onwards)
    - supressed[0.5:] (halfway through onwards)
    - ie+supressed[-1:] (last layer onwards plus input embeddings)
    - hs[-1]+supressed[-1:] (hs last layer plus last layer supressed)    
    """
    n = len(hidden_states)
    hs = rearrange(list(hidden_states), 'l b t h -> l b t h')
    To = hs.shape[2]
    Ti = inputs_embeds.shape[1]

    if '+' in method:
        methods = method.split('+')
    else:
        methods = [method]


    outs = []
    for method in methods:
        if '[' in method:
            # turn into slice
            spec = method.split('[')[1].split(']')[0]
            method = method.split('[')[0]
            lyr_slc = lloc(spec, n)
        else:
            lyr_slc = slice(None)
        
        if method == 'ie':
            # This extends into future tokens at times. Also there only one layer so no need to slice
            print(inputs_embeds.shape, hs.shape)
            x = inputs_embeds[:, :To][:, -1]
        elif method == 'hs':
            x = hs[lyr_slc, :, -1].sum(dim=0)
        elif method == 'supressed':
            supressed_act = get_supressed_activations(hs, w_out)[lyr_slc]
            x = reduce(supressed_act, 'l b t h -> b t h', 'sum')
            x = x[:, -1] # last token
            # x = repeat(x, 'b 1 h -> b t h', t=Ti)
        else:
            raise ValueError(f"Unknown method {method}")
        outs.append(x)
    
    # join the methods

    print(f"outs = {[o.shape for o in outs]}")
    o = outs[0]
    for i in range(1, len(outs)):
        o = o + outs[i]
    return o


# unit test
if __name__ == '__main__':
    # test hs2ie
    L, B, T, H = 4, 2, 3, 5
    hs = torch.randn(L, B, T, H) # l, b, t, h
    inputs_embeds = torch.randn(B, T, H) # b, t, h
    w_out = torch.randn(H, H) # h, h

    for method in ['hs[-1]', 'hs[0.5:]', 'hs[0.5]', 'supressed[-1:]', 'supressed[0.5:]', 'ie+supressed[-1:]', 'hs[-1]+supressed[-1:]']:
        print(f"method = {method}")
        spec = method.split('[')[1].split(']')[0]
        lyr_slc = lloc(spec, hs.shape[0])
        print(f"lyr_slc({method}) = {lyr_slc}")

        o = hs2ie(hs, inputs_embeds, w_out, method)
        print(f"o({method}) = {o.shape}")
        assert o.shape == (B, H), f"hs2ie({method}) = {o.shape} != (2, 5)"

