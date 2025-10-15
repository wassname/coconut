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


class TRMRecurser(nn.Module):
    """
    Tiny recursive reasoning module.
    
    Adapted from TinyRecursiveReasoningModel_ACTV1ReasoningModule.
    Simplified to just do fixed recursions without ACT.
    """
    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 8, expansion: float = 2.67):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, num_heads, expansion) 
            for _ in range(num_layers)
        ])

    def forward(self, latent_hs: torch.Tensor, context_hs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            latent_hs: [b, n_latents, hidden] - current latent hidden states
            context_hs: [b, seq, hidden] - context from question encoding (optional)
        
        Returns:
            refined_hs: [b, n_latents, hidden]
        """
        # Inject context (like TRM's input_injection)
        if context_hs is not None:
            # Simple mean pooling of context
            context_pooled = context_hs.mean(dim=1, keepdim=True)  # [b, 1, hidden]
            latent_hs = latent_hs + context_pooled
        
        # Apply transformer layers
        for layer in self.layers:
            latent_hs = layer(latent_hs)
        
        return latent_hs


class TRMTranscoder(nn.Module):
    """
    Transcodes hidden states → embedding space.
    
    Minimal MLP to bridge format mismatch between LLM hidden states and embeddings.
    """
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        self.proj = nn.Sequential(
            CastedLinear(hidden_size, int(hidden_size * expansion), bias=False),
            nn.GELU(),
            CastedLinear(int(hidden_size * expansion), hidden_size, bias=False),
        )
    
    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        return self.proj(hs)


class CoconutTRM(nn.Module):
    """
    TRM wrapper for Coconut.
    
    Manages:
    - Initialization of latent states
    - Detached recursions (no grad)
    - Recursions with gradient
    - Transcoding to embedding space
    """
    def __init__(
        self,
        hidden_size: int,
        n_latents: int = 4,
        n_detached: int = 2,
        num_layers: int = 2,
        num_heads: int = 8,
        expansion: float = 2.67,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_latents = n_latents
        self.n_detached = n_detached
        
        self.recurser = TRMRecurser(hidden_size, num_layers, num_heads, expansion)
        self.transcoder = TRMTranscoder(hidden_size, expansion=4.0)
        
        # Learnable initial state (like TRM's H_init, L_init)
        self.latent_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
    
    def forward(self, context_hs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_hs: [b, seq, hidden] - hidden states from LLM encoding question
        
        Returns:
            embeds: [b, n_latents, hidden] - embeddings ready for LLM decode
        """
        batch_size = context_hs.shape[0]
        
        # Initialize latent states
        latent_hs = self.latent_init.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.n_latents, self.hidden_size
        )
        
        # Detached recursions (accumulate junk, no grad)
        with torch.no_grad():
            for _ in range(self.n_detached):
                latent_hs = self.recurser(latent_hs, context_hs)
        
        # Recursions with gradient (learn to clean junk)
        for _ in range(self.n_latents - self.n_detached):
            latent_hs = self.recurser(latent_hs, context_hs)
        
        # Transcode to embedding space
        embeds = self.transcoder(latent_hs)
        
        return embeds
