import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from typing import Tuple, List, Union, Optional, Dict
from torch import Tensor
from collections import defaultdict


def calc_seq_vcr_loss_einops(
    hs: Float[Tensor, "b t h"],
    η: float = 1e-3,
    λ1: float = 1.0,
    λ2: float = 0.001,
    chunk_size: int = 16,
    cpu_offload: bool = False,
) -> Tuple[Float[Tensor, ""], Dict[str, float]]:
    """
    Einops + chunked variant of Seq-VCR loss.
      – chunk_size: number of timesteps to process at once (lower → lower peak GPU mem)
      – cpu_offload: move each covariance chunk to CPU before accumulating stats
    """
    B, T, P = hs.shape
    # (T, B, P)
    x = rearrange(hs, "b t h -> t b h")
    # mean per timestep: (T,1,P)
    x_mean = reduce(x, "t b h -> t 1 h", "mean")
    x_center = x - x_mean

    var_loss = torch.tensor(0.0, device=hs.device)
    cov_loss = torch.tensor(0.0, device=hs.device)

    for i in range(0, T, chunk_size):
        xs = x_center[i : i + chunk_size]  # (cs, B, P)
        # C_chunk: (cs, P, P)
        C_chunk = torch.einsum("cbh, cbd -> chd", xs, xs) / (B - 1)
        if cpu_offload:
            C_chunk = C_chunk.cpu()

        # variance term
        diag = torch.diagonal(C_chunk, dim1=-2, dim2=-1)  # (cs, P)
        var_loss += F.relu(1 - (diag + η).sqrt()).sum()

        # covariance term
        off = C_chunk - torch.diag_embed(diag)
        cov_loss += (off * off).sum()

    # normalize by total timesteps × features
    loss = (λ1 * var_loss + λ2 * cov_loss) / (T * P)
    metrics = {
        "loss_vcr_var": float(var_loss.detach()),
        "loss_vcr_cov": float(cov_loss.detach()),
    }
    return loss, metrics

def calc_seq_vcr_loss_vectorized(
    hs: Float[Tensor, "b t h"],
    η: float = 1e-3,
    λ1: float = 1.0,
    λ2: float = 0.001,
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
    var_loss = F.relu(1 - (diag + η).sqrt()).sum()

    # covariance term: mask out diag, square‐&‐sum
    mask = ~torch.eye(P, device=hs.device, dtype=torch.bool)  # (P, P)
    cov_loss = (C[..., mask].pow(2)).sum()

    # normalize
    loss = (λ1 * var_loss + λ2 * cov_loss) / (T * P)
    return loss, {
        "loss_vcr_var": float(var_loss),
        "loss_vcr_cov": float(cov_loss),
    }

# Test the loss function
def test_seq_vcr_loss():
    # Create random tensor for testing
    batch_size, seq_len, hidden_dim = 16, 100, 256
    hs = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Calculate loss using both implementations
    loss1, metrics1 = calc_seq_vcr_loss_vectorized(hs)
    loss2, metrics2 = calc_seq_vcr_loss_einops(hs)
    
    print("Standard implementation:")
    print(f"  Total loss: {loss1.item():.6f}")
    print(f"  Variance loss: {metrics1['loss_vcr_var']:.6f}")
    print(f"  Covariance loss: {metrics1['loss_vcr_cov']:.6f}")
    
    print("\nEinops implementation:")
    print(f"  Total loss: {loss2.item():.6f}")
    print(f"  Variance loss: {metrics2['loss_vcr_var']:.6f}")
    print(f"  Covariance loss: {metrics2['loss_vcr_cov']:.6f}")
    
    # Check if both implementations give similar results
    assert torch.isclose(loss1, loss2, rtol=1e-4), "Implementations give different results"
    print("\nBoth implementations give similar results! ✓")

if __name__ == "__main__":
    test_seq_vcr_loss()

import torch
from coconut.vcr_loss import VCRLoss

B, T, P = 16, 50, 512
# fake per‐layer outputs: list of (B,T,P) tensors
hs_l = [torch.randn(B, T, P, device='cuda') for _ in range(2)]

# Test hidden sizes
for D in [1,2,4,8,16,32,64,128]:
    vcr = VCRLoss(H=P, D=D).cuda()
    loss, metrics = vcr(hs_l)
    print(f"D={D:3d}  total={loss.item():.4f}  var={metrics['loss_vcr_var']:.2f}  cov={metrics['loss_vcr_cov']:.2f}")

# Now test token length
for T in [1,2,4,8,16,32,64,128]:
    hs_l = [torch.randn(B, T, P, device='cuda') for _ in range(2)]
    vcr = VCRLoss(H=P, D=32).cuda()
    loss, metrics = vcr(hs_l)
    print(f"T={T:3d}  total={loss.item():.4f}  var={metrics['loss_vcr_var']:.2f}  cov={metrics['loss_vcr_cov']:.2f}")
