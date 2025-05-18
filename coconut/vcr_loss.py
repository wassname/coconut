from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from typing import Tuple, List, Union, Optional, Dict
from torch import Tensor
import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F


def calc_seq_vcr_loss(
    hs: Float[Tensor, "b t h"],
    η: float = 1e-3,
    λ1: float = 0.001,
    λ2: float = 0.0001,
) -> Tuple[Float[Tensor, ""], Dict[str, float]]:
    """
    Fully-vectorized Seq-VCR loss: no Python for‐loop.
    """
    B, T, P = hs.shape
    # (T, B, P)
    x = rearrange(hs, "b t h -> t b h")
    x_center = x - x.mean(dim=1, keepdim=True)  # center per timestep

    # full covariance tensor: (T, P, P)
    C = torch.einsum("t b p, t b q -> t p q", x_center, x_center) / (B - 1)

    # variance term: sqrt(diag + η)
    diag = torch.diagonal(C, dim1=1, dim2=2)             # (T, P)
    var_loss = F.relu(1 - (diag + η).sqrt()).sum() / (T * P)

    # covariance term: mask out diag, square‐&‐sum
    mask = ~torch.eye(P, device=hs.device, dtype=torch.bool)  # (P, P)
    cov_loss = (C[..., mask].pow(2)).sum() / (T * P * (P - 1)) # (P, P)
    # Note this is different from the original code, which normalizes by T * P. But this means we don't have to keep changing λ2 like in the paper, so it seems better to me. The downside is we don't regularise large hidden spaces more, but changing λ2 shows that didn't work anyway.

    # normalize
    loss = (λ1 * var_loss + λ2 * cov_loss)
    return loss, {
        "loss_vcr_var": λ1 * var_loss.item(),
        "loss_vcr_cov": λ2 * cov_loss.item(),
    }

class VCRLoss(nn.Module):
    def __init__(self, H=1536, D=32):
        super().__init__()

        # Fixed random projection
        self.down_proj = nn.Linear(
            H, D, bias=False
        )
        nn.init.orthogonal_(self.down_proj.weight)
        self.down_proj.weight.requires_grad = False
        # FIXME we should be applying this to the final high dim output too, but wait for reference implementation https://github.com/rarefin/SEQ_VCR

    def forward(self, hs_l: Float[Tensor, "l b t h"]) -> Float[Tensor, "b"]:
        # for each layer
        loss = 0
        logs = defaultdict(float)
        for hs in hs_l:
            hs2 = self.down_proj(hs)
            loss_i, extra = calc_seq_vcr_loss(hs2)
            for k, v in extra.items():
                logs[k] += v
            loss += loss_i
            torch.cuda.empty_cache()

        return loss, dict(logs)
