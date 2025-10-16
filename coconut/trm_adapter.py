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
from .trm_layers import Attention, SwiGLU, rms_norm, CastedLinear


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
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, expansion: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, num_heads, expansion) for _ in range(num_layers)
        ])
        self.context_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(self, zL, zH, context_hs):
        # Inject context and high-level state
        context_pooled = context_hs.mean(dim=1, keepdim=True)
        in_state = zL + zH + self.context_proj(context_pooled)
        
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

    def forward(self, zH, zL):
        in_state = zH + zL
        for layer in self.layers:
            in_state = layer(in_state)
        return in_state

class TRMTranscoder(nn.Module):
    """Transcodes the final high-level state zH into the LLM's embedding space."""
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        self.proj = nn.Sequential(
            CastedLinear(hidden_size, int(hidden_size * expansion), bias=False),
            nn.GELU(),
            CastedLinear(int(hidden_size * expansion), hidden_size, bias=False),
        )
    
    def forward(self, zH: torch.Tensor) -> torch.Tensor:
        return self.proj(zH)

class CoconutTRM(nn.Module):
    """
    HRM-style wrapper for Coconut, implementing the dual-network recursive
    model matching the paper's pseudocode.
    """
    def __init__(
        self,
        hidden_size: int,
        trm_n_sup: int,
        n_detached_recursions: int,
        n_gradient_recursions: int,
        num_layers: int,
        num_heads: int,
        expansion: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_sup = trm_n_sup
        self.n_detached = n_detached_recursions
        self.n_gradient = n_gradient_recursions
        
        self.l_net = L_net(hidden_size, num_layers, num_heads, expansion)
        self.h_net = H_net(hidden_size, num_layers, num_heads, expansion)
        self.transcoder = TRMTranscoder(hidden_size, expansion=4.0)
        
        # Learnable initial states for zL and zH
        self.zL_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.zH_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        
    def hrm(self, zL, zH, context_hs, max_loops: int=100) -> tuple:
        """
        Single HRM recursion step matching the paper's hrm() function.
        Does n_detached recursions without gradients, then n_gradient with gradients.
        During training, computes EMA of zH over gradient recursions for deep supervision proxy.
        """
        # Add sequence dimension for processing
        zL_step, zH_step = zL.unsqueeze(1), zH.unsqueeze(1)

        # Detached recursions
        with torch.no_grad():
            for _ in range(self.n_detached):
                zL_step = self.l_net(zL_step, zH_step, context_hs)
                zH_step = self.h_net(zH_step, zL_step)

        ema_decay = 0.9  # tunable

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
            
        return zL_step.squeeze(1), zH_step.squeeze(1), ema_zH
        
    def forward(self, context_hs: torch.Tensor, zL_prev=None, zH_prev=None, max_loops=100) -> tuple:
        """
        Performs one HRM step (not the full deep supervision loop).
        Returns the embedding for LLM decoder and updated latent states.
        Uses EMA zH during training for supervision, final zH during inference.
        """
        batch_size = context_hs.shape[0]
        
        # Initialize latent states if not provided
        if zL_prev is None:
            zL = self.zL_init.unsqueeze(0).expand(batch_size, -1)
        else:
            zL = zL_prev
            
        if zH_prev is None:
            zH = self.zH_init.unsqueeze(0).expand(batch_size, -1)
        else:
            zH = zH_prev

        # Run one HRM recursion
        zL_next, _, zH_next = self.hrm(zL, zH, context_hs, max_loops=max_loops)
        

        latent_embed = self.transcoder(zH_next)
        return latent_embed.squeeze(1), zL_next, zH_next
