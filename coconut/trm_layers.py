# Copied from docs/trm_reference_code/models/layers.py
# Minimal dependencies for TRM recursion

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

CosSin = Tuple[torch.Tensor, torch.Tensor]


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, num_heads, seq_len, head_dim] or similar
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(1 if q.ndim == 3 else (0,1))) + (rotate_half(q) * sin.unsqueeze(1 if q.ndim == 3 else (0,1)))
    k_embed = (k * cos.unsqueeze(1 if k.ndim == 3 else (0,1))) + (rotate_half(k) * sin.unsqueeze(1 if k.ndim == 3 else (0,1)))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Parameter(emb.cos(), requires_grad=False)
        self.sin_cached = nn.Parameter(emb.sin(), requires_grad=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Simple Xavier init
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / (in_features ** 0.5)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), 
                       bias=self.bias.to(input.dtype) if self.bias is not None else None)


class SwiGLU(nn.Module):
    """SwiGLU activation from TRM."""
    def __init__(self, hidden_size: int, expansion: float = 2.67):
        super().__init__()
        round_to = 16
        inter = int(round(expansion * hidden_size * 2 / 3 / round_to)) * round_to  # round to 16
        assert inter > hidden_size, "Expansion must be > 1.5"
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    """Simplified attention from TRM (with optional RoPE)."""
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, 
                                     (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, 
                                     bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        # Reshape for attention
        query = query.transpose(1, 2)  # [b, h, s, d]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        if cos_sin is not None:
            cos, sin = cos_sin
            # Assume cos/sin are [seq_len, head_dim//2 * 2]; broadcast to [1, seq_len, head_dim] if needed
            if sin.shape[0] != seq_len:
                cos = cos[:seq_len]
                sin = sin[:seq_len]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Use PyTorch's scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            query=query, key=key, value=value, is_causal=self.causal
        )
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)
