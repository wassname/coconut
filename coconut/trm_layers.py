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
        inter = int(round(expansion * hidden_size * 2 / 3 / 256)) * 256  # round to 256
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    """Simplified attention from TRM (no RoPE for now)."""
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
        
        # Use PyTorch's scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            query=query, key=key, value=value, is_causal=self.causal
        )
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)
