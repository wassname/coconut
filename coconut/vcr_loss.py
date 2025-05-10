from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from typing import Tuple, List, Union, Optional, Dict
from torch import Tensor
import torch
from collections import defaultdict
import torch.nn as nn


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
