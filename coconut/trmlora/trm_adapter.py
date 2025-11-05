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
from loguru import logger
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
from .trm_layers import Attention, SwiGLU, rms_norm, CastedLinear, trunc_normal_init_
import wandb
from einops import repeat, rearrange

hs_bsh = Float[Tensor, 'b s h']
hs_b1h = Float[Tensor, 'b 1 h']
z_bh = Float[Tensor, 'b h']
z_b1h = Float[Tensor, 'b 1 h']


class TRMBlock(nn.Module):
    """Single TRM transformer block (from original TRM).
    
    https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9/models/recursive_reasoning/trm.py#L65
    """
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
    """
    Low-level recursive reasoning module (HRM).

    from https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9/models/recursive_reasoning/trm.py#L65
    """
    

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, expansion: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, num_heads, expansion) for _ in range(num_layers)
        ])

    def forward(self, hidden_states: z_b1h, input_injection: z_b1h, **kwargs) -> z_b1h:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


def trm_seq(
    l_net: L_net,
    zL: Float[Tensor, 'b r'],
    zH: Float[Tensor, 'b r'], 
    x: Float[Tensor, 'b s r'],
    latent_mask: Bool[Tensor, 'b s'],
    l_cycles: int,
    h_cycles: int,
) -> tuple[Float[Tensor, 'b s r'], Float[Tensor, 'b s r']]:
    """Sequence-level TRM refinement.
        
    """
    # We inject x into the latent states, if it was unnormalised it might be destabilizing and the model would have to learn to ignore it which it can't do easily since it's directly added
    x = rms_norm(x)

    if h_cycles < 1:
        return zL, zH
    
    b, s, r = x.shape

    # Here we recurse along the sequence of tokens, only where the latent mask is set
    zH_out = []
    zL_out = []

    for i in range(s):
        mask_i = latent_mask[:, i]  # [b]
        if mask_i.any():
            # Process subset with latents: zH[mask_i] is [n_latent, r]
            zH_deep, zL_deep = trm_inner(
                l_net=l_net,
                zH=zH[mask_i, None], zL=zL[mask_i, None], x=x[mask_i, i:i+1, :],  # [n_latent, 1, r]
                l_cycles=l_cycles, h_cycles=h_cycles
            )
            # Scatter results back into full batch (gradient-safe, creates new tensor)
            indices = mask_i.nonzero(as_tuple=False).squeeze(-1)  # [n_latent]
            idx_expanded = indices.unsqueeze(-1).expand(-1, r).to(zH.device)  # [n_latent, r]
            zH = zH.scatter(0, idx_expanded, zH_deep.squeeze(1))  # [b, r] <- [n_latent, r]
            zL = zL.scatter(0, idx_expanded, zL_deep.squeeze(1))

        zH_out.append(zH)
        zL_out.append(zL)

    zL = torch.stack(zL_out, dim=1)  # [b, s, r]
    zH = torch.stack(zH_out, dim=1)  # [b, s, r]

    return zL, zH



def trm_inner(
    l_net: L_net,
    zL: Float[Tensor, 'bs 1 r'],
    zH: Float[Tensor, 'bs 1 r'], 
    x: Float[Tensor, 'bs s r'],
    l_cycles: int,
    h_cycles: int,
) -> tuple[Float[Tensor, 'bs 1 r'], Float[Tensor, 'bs 1 r']]:
    """
    Tiny Recursion Module (TRM) core logic.
    
    Gradient flow: Early H cycles run no_grad (detached), final cycles keep grad.
    When added to base_hidden (which has grad), detached recursions act as leaf nodes,
    allowing model to learn error cleanup from its own accumulated mistakes (see TRM paper).
    
    Args:
        l_net: L_net module for recursive refinement
        zL: Latent reasoning state [b, r]
        zH: High-level output state [b, r]
        context: Context from down-projection [b, s, r]
        l_cycles: Number of L_net cycles per H cycle
        h_cycles: Number of H cycles
        
    Returns:
        (zL_refined, zH_refined) both [b, s, r]
    """

    def latent_recursion(x, zH, zL, n=1):
        for _ in range(n):  # latent reasoning with context
            zL = l_net(zL, x + zH)
        zH = l_net(zH, zL)  # refine output answer
        return zH, zL

    # Early H cycles detached: forms leaf nodes but gradients still flow via base_hidden trunk and also via `context`
    with torch.no_grad():
        for _ in range(max(0, h_cycles - 1)):
            zH, zL = latent_recursion(x, zH, zL, n=l_cycles)
    
    # Final cycle with grad
    zH, zL = latent_recursion(x, zH, zL, n=l_cycles)

    return zL, zH
