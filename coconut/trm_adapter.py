"""
TRM Adapter for Coconut

Minimal adaptation of TRM (Tiny Recursive Model) for use with frozen LLMs.
Based on docs/trm_reference_code/models/recursive_reasoning/trm.py

Key simplifications from original TRM:
- No ACT (Adaptive Computation Time) - just fixed recursions
- No Q-learning for halting
- No hierarchical H/L split - just one recursive module
- Direct integration with Coconut's latent token framework
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
from .trm_layers import Attention, SwiGLU, rms_norm, CastedLinear

hs_bsh = Float[Tensor, 'b s h']
hs_b1h = Float[Tensor, 'b 1 h']
z_bh = Float[Tensor, 'b h']
z_b1h = Float[Tensor, 'b 1 h']


class TRMBlock(nn.Module):
    """Single TRM transformer block (from original TRM)."""
    def __init__(self, hidden_size: int, num_heads: int, expansion: float = 2.67, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_eps = rms_norm_eps
        
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm architecture from TRM
        # Self attention + residual + norm
        hidden_states = rms_norm(
            hidden_states + self.self_attn(hidden_states), 
            variance_epsilon=self.norm_eps
        )
        # MLP + residual + norm
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps
        )
        return hidden_states


class L_net(nn.Module):
    """Low-level recursive reasoning module (HRM)."""
    def __init__(self, hidden_size: int, llm_hidden_size: int, num_layers: int, num_heads: int, expansion: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, num_heads, expansion) for _ in range(num_layers)
        ])
        self.context_proj = CastedLinear(llm_hidden_size, hidden_size, bias=False)

    def forward(self, zL: z_b1h, zH: z_b1h, context_hs: hs_bsh) -> z_b1h:
        # Inject context and high-level state
        # # TODO update and remove - should apply attention and weighted, or just use latest? mean?. Try latest...
        # context_pooled = context_hs.mean(dim=1, keepdim=True)
        last_hs = context_hs[:, -1:, :]
        in_state = zL + zH + self.context_proj(last_hs)
        
        for layer in self.layers:
            in_state = layer(in_state)
        return in_state

class H_net(nn.Module):
    """High-level recursive reasoning module (HRM)."""
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, expansion: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, num_heads, expansion) for _ in range(num_layers)
        ])

    def forward(self, zH: z_b1h, zL: z_b1h) -> z_b1h:
        in_state = zH + zL
        for layer in self.layers:
            in_state = layer(in_state)
        return in_state

class TRMTranscoder(nn.Module):
    """Transcodes the final high-level state zH into the LLM's embedding space."""
    def __init__(self, hidden_size: int, llm_hidden_size: int, expansion: float = 1.0, trm_transcoder_layers: int = 1, llm_embed: torch.Tensor = None):
        super().__init__()

        layers = [
            CastedLinear(hidden_size, int(hidden_size * expansion), bias=False),
        ]
        for _ in range(trm_transcoder_layers - 1):
            layers.append(nn.GELU())
            layers.append(CastedLinear(int(hidden_size * expansion), int(hidden_size * expansion), bias=False))
        self.proj = nn.Sequential(*layers)

        # In TRMTranscoder.__init__
        nn.init.xavier_uniform_(self.proj[2].weight)
        self.proj[2].weight.data *= 0.01  # scale down 100x

        # TODO or consider init from LLM embedding matrix using SVD low-rank approx, this will help it transcode to the LLM space better.
        self.svd_initialized = False
        self.init_svd(llm_embed)
    
    def forward(self, zH: z_bh) -> hs_b1h:
        return self.proj(zH)
    
    def init_svd(self, llm_embed: torch.Tensor):
        """Initialize transcoder final layer using low-rank approx of LLM embed matrix."""
        if self.svd_initialized:
            return
        # Low-rank approx of embed (vocab x h_llm) to get h_trm x h_llm projection
        rank = min(512, self.hidden_size)
        U, S, Vh = torch.svd_lowrank(llm_embed, q=rank)
        low_rank_proj = U @ torch.diag(S) @ Vh  # h_llm x h_llm approx
        # Take first h_trm rows to get h_trm x h_llm
        with torch.no_grad():
            self.transcoder[-1].weight.copy_(low_rank_proj[:self.hidden_size, :] * 0.1)
        self.svd_initialized = True

class CoconutTRM(nn.Module):
    """
    Configurable TRM/HRM-style wrapper for Coconut, with optional dual/single net and transcoder layers.
    """
    def __init__(
        self,
        hidden_size: int,
        llm_hidden_size: int,
        trm_h_layers: int,
        trm_l_layers: int,
        trm_h_cycles: int,
        trm_l_cycles: int,
        num_heads: int,
        expansion: float,
        trm_transcoder_layers: int,
        # n_detached_recursions: int,
        # n_gradient_recursions: int,
        llm_embed: torch.Tensor = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.trm_h_layers = trm_h_layers
        self.trm_l_layers = trm_l_layers
        self.trm_h_cycles = trm_h_cycles
        self.trm_l_cycles = trm_l_cycles
        self.num_heads = num_heads
        self.expansion = expansion
        # self.trm_transcoder_layers = trm_transcoder_layers
        # self.n_detached = n_detached_recursions
        # self.n_gradient = n_gradient_recursions
        
        # L_net always present
        self.l_net = L_net(hidden_size, self.llm_hidden_size, self.trm_l_layers, num_heads, expansion)

        self.h_net = H_net(hidden_size, self.trm_h_layers, num_heads, expansion)
        
        # Configurable transcoder with SwiGLU layers
        self.transcoder = TRMTranscoder(hidden_size, llm_hidden_size, expansion=expansion, trm_transcoder_layers=trm_transcoder_layers, 
                                        llm_embed=llm_embed)
        
        # Learnable initial states
        self.zL_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.zH_init = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def hrm(self, zL: z_bh, zH: z_bh, context_hs: hs_bsh) -> tuple:
        """
        Configurable recursion step.
        For dual net (h_layers >0): T-1 detached full cycles (n L_steps + 1 H_step), then 1 full with grad.
        For single net (h_layers=0): Use l_net for both, with input injection for z vs y update.
        Computes EMA of zH during grad part for deep supervision proxy.
        """
        # Add sequence dimension
        zL_step, zH_step = zL.unsqueeze(1), zH.unsqueeze(1)

        # Detached recursions
        with torch.no_grad():
            for H_step in range(self.config.H_cycles-1):
                # FIXME in TRM l_net and h_net are the same layer with swapped inputs
                for L_step in range(self.config.L_cycles):
                    zL_step = self.l_net(zL_step, zH_step, context_hs)
                zH_step = self.h_net(zH_step, zL_step)

        ema_decay = 0.9 # TODO add to config

        if self.training:
            # why do we do EMA? this is our proxy for deep supervision. Deep supervision would be too expensive to compute requiring LLM forward passes for each. So instead we use EMA to get a smoothed version of the final latent state. This means we can provide partial supervision to intermediate states, hopefully preventing collapse.
            
            # Gradient recursions with EMA
            ema_zH = zH_step.clone()  # init EMA
            for _ in range(min(max_loops//2, self.n_gradient)):
                zL_step = self.l_net(zL_step, zH_step, context_hs)
                zH_step = self.h_net(zH_step, zL_step)
                ema_zH = ema_decay * ema_zH + (1 - ema_decay) * zH_step
        else:
            # Gradient recursions without EMA
            for _ in range(min(max_loops//2, self.n_gradient)):
                zL_step = self.l_net(zL_step, zH_step, context_hs)
                zH_step = self.h_net(zH_step, zL_step)
            ema_zH = zH_step  # use final for inference
            
        return zL_step.squeeze(1), zH_step.squeeze(1), ema_zH.squeeze(1)

    def forward(self, context_hs: hs_bsh, zL_prev: Optional[z_bh] = None, zH_prev: Optional[z_bh] = None) -> tuple:
        """
        Performs one configurable recursion step.
        Returns the embedding for LLM decoder and updated latent states.
        Uses EMA zH during training for supervision, final zH during inference.
        """
        batch_size = context_hs.shape[0]
        
        # Initialize latent states if not provided
        if zL_prev is None:
            zL = self.zL_init.unsqueeze(0).expand(batch_size, -1)
        else:
            zL = zL_prev
            assert zL.ndim == 2
            
        if zH_prev is None:
            zH = self.zH_init.unsqueeze(0).expand(batch_size, -1)
        else:
            zH = zH_prev
            assert zH.ndim == 2

        # Run one HRM recursion
        zL_next, _, zH_next = self.hrm(zL, zH, context_hs)
        

        diff_to_hs = self.transcoder(zH_next)
        return diff_to_hs.squeeze(1), zL_next, zH_next
