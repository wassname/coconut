import time
import math
from typing import Literal, Optional, Tuple, List
import torch
from torch import Tensor, nn
import numpy as np
from tqdm import tqdm
import nn.functional as F
from torch_sparse import SparseTensor, transpose
from einops import rearrange, repeat
from jaxtyping import Float, Int, Bool


def create_orthonormal_matrix(A):
    # returns an orthonormal matrix (square) of size (min(A.shape), min(A.shape))
    Q, R = torch.qr(A)
    return Q


def get_target_modules_list(model, target_modules):
    target_names = []
    for n, _ in model.named_modules():
        if any(t in n for t in target_modules):
            target_names.append(n)
    return target_names


def replace_svft_with_fused_linear(model, target_modules_list):
    print("Replacing SVFT layers with new Linear layers")

    # filter out svft layer
    target_modules_list = [l for l in target_modules_list if "svft_layer" not in l]

    for target_path in tqdm(
        reversed(target_modules_list), total=len(target_modules_list)
    ):
        parent_path = (
            target_path[: target_path.rfind(".")] if "." in target_path else ""
        )
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        in_dim = target.svft_layer.v.shape[1]
        out_dim = target.svft_layer.u.shape[0]
        if target.bias is None:
            lin = nn.Linear(in_dim, out_dim, bias=False)
        else:
            lin = nn.Linear(in_dim, out_dim, bias=True)
            lin.bias.data = target.bias.data
        lin.weight.data = target.merge_and_unload()
        parent.__setattr__(target_name, lin)


def create_and_replace_modules(model, target_modules_list, create_fn):
    print("Replacing Linear layers with SVFT layers")

    for target_path in tqdm(
        reversed(target_modules_list), total=len(target_modules_list)
    ):
        parent_path = (
            target_path[: target_path.rfind(".")] if "." in target_path else ""
        )
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        parent.__setattr__(target_name, create_fn(target))


class SVFTLayer(nn.Module):
    def __init__(
        self,
        u,
        s,
        v,
        off_diag=0,
        pattern: Literal["banded", "random", "top_k"] = "banded",
        r=None,
        fill_orthonormal=False,
    ):
        """
        @inputs:
            u: torch.Tensor. Left singular vectors of pre-trained weight matrix
            s: torch.Tensor. Singular values of pre-trained weight matrix
            v: torch.Tensor. Right singular vectors of pre-trained weight matrix
            off_diag: int. Total off-diagonals to be used to populate matrix M (as referred in main paper)
            pattern: str. Choices: "banded", "random", "top_k". Using "banded" with off_diag=1 simulates SVFT-plain
            r: int. Rank constrains how many singular vectors and values to use.
            fill_orthonormal: bool. To determine if random orthonormal basis should be used
        """

        super().__init__()

        self.off_diag = off_diag
        r = s.shape[0] if r is None else min(s.shape[0], r)
        self.r = r
        diff_rank = s.shape[0] - r

        # Crop U V S to rank
        if fill_orthonormal:
            Q_u = torch.randn_like(u).to(s.device)
            nn.init.orthogonal_(Q_u)
            Q_v = torch.randn_like(v).to(s.device)
            nn.init.orthogonal_(Q_v)

            u = torch.cat([u[:, :r], Q_u[:, :diff_rank]], dim=1)
            v = torch.cat([v[:r, :], Q_v[:diff_rank, :]], dim=0)
            s = torch.cat([s[:r], torch.zeros(diff_rank).to(s.device)], dim=0)
            r = self.r = s.shape[0]

        else:
            s = s[:r]
            u = u[:, :r]
            v = v[:r, :]

        # ingredients for sparse s0
        s0 = s.cpu().detach().clone().contiguous()
        self.s0 = nn.Parameter(s0, requires_grad=False)
        self.s0_edge_index = (
            torch.sparse.spdiags(s0, torch.LongTensor([0]), (r, r)).coalesce().indices()
        )
        self.register_buffer("s0_row", self.s0_edge_index[0])
        self.register_buffer("s0_col", self.s0_edge_index[1])

        if pattern == "random":
            print("Random pattern")
            k = r * (2 * self.off_diag + 1) - self.off_diag * (self.off_diag + 1)
            rows = torch.randint(0, r, (k,))
            cols = torch.randint(0, r, (k,))
            self.s_edge_index = torch.stack([rows, cols])

        elif pattern == "banded":
            diags = 2 * self.off_diag + 1
            offsets_positive = torch.arange(0, self.off_diag + 1)
            offsets_negative = torch.arange(-1, -self.off_diag - 1, -1)
            self.offsets = torch.cat([offsets_positive, offsets_negative])
            self.s_edge_index = (
                torch.sparse.spdiags(torch.randn([diags, r]), self.offsets, (r, r))
                .coalesce()
                .indices()
            )
            k = self.s_edge_index.shape[1]
        elif pattern == "top_k":
            if u.shape == v.shape:
                coeffs = u @ v.T
            else:
                coeffs = u if u.shape[0] == u.shape[1] else v

            k = r * (2 * self.off_diag + 1) - self.off_diag * (self.off_diag + 1)
            # Flatten the tensor to 1D
            flattened_tensor = coeffs.contiguous().view(-1)
            _, top_indices_flat = torch.topk(flattened_tensor, k)
            num_rows, num_cols = coeffs.size()
            rows = top_indices_flat // num_cols
            cols = top_indices_flat % num_cols
            self.s_edge_index = torch.stack([rows, cols])

        
        # our two learnable parameters are sd and gate
        self.gate = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32), requires_grad=True
        )
        # sd is sparse
        self.sd = nn.Parameter(torch.zeros(k), requires_grad=True)
        nn.init.kaiming_normal_(self.sd[None, :])
        self.register_buffer("sd_row", self.s_edge_index[0])
        self.register_buffer("sd_col", self.s_edge_index[1])


        self.u = nn.Parameter(u.clone().detach().contiguous(), requires_grad=False)
        self.v = nn.Parameter(v.clone().detach().contiguous(), requires_grad=False)

    def forward(self, x: Float[Tensor, "b i"]) -> Float[Tensor, "b o"]:
        V, U = self.v, self.u
        s_eff = self.get_sparse_s_eff()
        if self.training:
            x = (x @ V.T) @ s_eff.T @ U.T
        else:
            x = x @ self.get_weights()
        return x

    def get_sparse_s_eff(self) -> Float[Tensor, "r r"]:
        sd = SparseTensor(
            row=self.sd_row, col=self.sd_col, value=self.sd * F.sigmoid(self.gate)
        )
        s0 = SparseTensor(row=self.s0_row, col=self.s0_col, value=self.s0)

        # TODO also add a multiplicative mode
        # TODO consider a delora style lambda regularization https://github.com/ExplainableML/DeLoRA http://r.jina.ai/https://arxiv.org/pdf/2503.18225
        s_eff = s0 + sd
        return s_eff

    def get_weights(self):
        V, U = self.v, self.u
        s_eff = self.get_sparse_s_eff()
        W = (s_eff @ V).T @ U.T
        return W

    def merge_and_unload(self):
        return self.get_weights().T.contiguous()


class LinearWithSVFT(nn.Module):
    def __init__(
        self, linear, off_diag, pattern="banded", rank=None, fill_orthonormal=False
    ):
        """
        @inputs:
                linear: torch.Tensor. Linear Layer that has to adapted
                off_diag: int. total number off diagonals to be used if pattern is 'banded'
                          for remaining patterns, equivalent number of learnable parameters are learnt
                rank: SVD rank
                fill_orthonormal: bool. To determine if random orthonormal basis should be used
        """

        super().__init__()

        self.bias = linear.bias

        # since linear.weight is on GPU, computing SVD will be significantly faster
        svd = torch.linalg.svd(linear.weight, full_matrices=False)

        self.svft_layer = SVFTLayer(
            svd[0],
            svd[1],
            svd[2],
            off_diag=off_diag,
            pattern=pattern,
            r=rank,
            fill_orthonormal=fill_orthonormal,
        )

    def forward(self, x):
        if self.bias is not None:
            return self.svft_layer(x) + self.bias

        else:
            return self.svft_layer(x)

    def merge_and_unload(self):
        return self.svft_layer.merge_and_unload()


def freeze_model(model, exclude_list=None):
    """Freeze all parameters of the model"""
    if exclude_list is None:
        exclude_list = []

    for n, p in model.named_parameters():
        if not any(e in n for e in exclude_list):
            p.requires_grad = False
