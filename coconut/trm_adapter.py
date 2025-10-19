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
    """Transcodes TRM's latent state zH into LLM's hidden space.
    
    Architecture: zH [h_trm] -> MLP [h_trm*expansion] -> final_proj [h_llm]   

    """
    def __init__(self, hidden_size: int, llm_hidden_size: int, expansion: float = 1.0, trm_transcoder_layers: int = 1, llm_embed: torch.Tensor = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size

        # MLP in TRM's latent space
        layers = []
        for _ in range(trm_transcoder_layers):
            layers.append(CastedLinear(hidden_size, int(hidden_size * expansion), bias=False))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Final projection to LLM space
        input_dim = int(hidden_size * expansion) if trm_transcoder_layers > 0 else hidden_size
        self.final_proj = CastedLinear(input_dim, llm_hidden_size, bias=False)
        nn.init.xavier_uniform_(self.final_proj.weight, gain=0.01)

        # Optional SVD-based initialization
        self.svd_initialized = False
        self.init_svd(llm_embed)
    
    def forward(self, zH: z_bh) -> hs_b1h:
        features = self.mlp(zH)  # [b, h_trm*expansion]
        output = self.final_proj(features)  # [b, h_llm]
        return output.unsqueeze(1)  # [b, 1, h_llm]
    
    def init_svd(self, llm_embed: Float[Tensor, 'vocab h_llm']):
        """Initialize final_proj using SVD basis of LLM embed matrix as prior.
        
        Logic:
        - LLM embedding matrix [vocab, h_llm] maps tokens to semantic space
        - SVD extracts principal components: Vh columns are the "semantic directions"
        - Use these as initial weights to bias transcoder toward embedding-like outputs
        - final_proj weight is [h_llm, input_dim] for nn.Linear
        - We want each input_dim feature to map to a combo of semantic directions
        - So we use Vh [h_llm, rank] as the basis, pad/slice to match input_dim
        """
        if self.svd_initialized or llm_embed is None:
            return
        
        # SVD of embedding matrix: vocab x h_llm -> rank principal components
        rank = min(512, llm_embed.shape[0], llm_embed.shape[1])
        U, S, Vh = torch.svd_lowrank(llm_embed.float(), q=rank)
        # Vh is [h_llm, rank] - columns are the principal semantic directions
        
        # Target weight is [h_llm, input_dim] for nn.Linear
        target_shape = self.final_proj.weight.shape  # [h_llm, input_dim]
        
        with torch.no_grad():
            # Vh is already [h_llm, rank], no transpose needed
            basis = Vh  # [h_llm, rank]
            
            # Pad or slice to match input_dim
            if rank >= target_shape[1]:  # input_dim
                # Slice top input_dim components
                init_weight = basis[:, :target_shape[1]]  # [h_llm, input_dim]
            else:
                # Pad with small random if rank < input_dim
                init_weight = torch.zeros(target_shape, dtype=basis.dtype, device=basis.device)
                init_weight[:, :rank] = basis  # [h_llm, rank]
                init_weight[:, rank:] = torch.randn_like(init_weight[:, rank:]) * 0.01
            
            # Scale down to avoid dominating early training
            self.final_proj.weight.copy_(init_weight * 0.1)
            
        self.svd_initialized = True
        logger.info(f"Initialized transcoder with SVD basis: rank={rank}, weight_shape={target_shape}")


class CoconutTRM(nn.Module):
    """
    Configurable TRM/HRM-style wrapper for Coconut, with optional dual/single net and transcoder layers.
    """
    def __init__(
        self,
        hidden_size: int,
        llm_hidden_size: int,
        # trm_h_layers: int,
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
        self.forward_dtype = torch.bfloat16
        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size
        # self.h_layers = trm_h_layers
        # self.l_layers = trm_l_layers
        self.h_cycles = trm_h_cycles
        self.trm_l_cycles = trm_l_cycles
        self.num_heads = num_heads
        self.expansion = expansion
        # self.trm_transcoder_layers = trm_transcoder_layers
        # self.n_detached = n_detached_recursions
        # self.n_gradient = n_gradient_recursions
        
        # L_net always present
        self.l_net = L_net(hidden_size, llm_hidden_size, trm_l_layers, num_heads, expansion)

        # self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)]) # TODO rm ref
        # self.h_net = H_net(hidden_size, self.trm_h_layers, num_heads, expansion) # TODO remove no hnet
        
        # Configurable transcoder with SwiGLU layers
        self.transcoder = TRMTranscoder(hidden_size, llm_hidden_size, expansion=expansion, trm_transcoder_layers=trm_transcoder_layers, 
                                        llm_embed=llm_embed)
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # self.lm_head      = CastedLinear(hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(hidden_size, 2, bias=True)
        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore
        
    def hrm(self, zL: z_bh, zH: z_bh, context_hs: hs_bsh) -> tuple:
        """
        Configurable recursion step.
        For dual net (h_layers >0): T-1 detached full cycles (n L_steps + 1 H_step), then 1 full with grad.
        For single net (h_layers=0): Use l_net for both, with input injection for z vs y update.\
        """
        # Add sequence dimension
        zLs, zHs = zL.unsqueeze(1), zH.unsqueeze(1)

        # H_cycles-1 without grad
        with torch.no_grad():
            for H_step in range(self.h_cycles-1):
                for L_step in range(self.l_cycles):
                    zLs = self.l_net(zLs, zHs + context_hs)
                zHs = self.l_net(zHs, zLs)

        # 1 with grad
        for L_step in range(self.l_cycles):
            zLs = self.l_net(zLs, zHs + context_hs)
        zHs = self.l_net(zHs, zLs)

        return zHs.squeeze(1), zLs.squeeze(1)

    def forward(self, context_hs: hs_bsh, zL_prev: Optional[z_bh] = None, zH_prev: Optional[z_bh] = None) -> tuple:
        """
        Performs one configurable recursion step.
        Returns the embedding for LLM decoder and updated latent states.
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
        zL_next, zH_next = self.hrm(zL, zH, context_hs)

        # LM outputs
        # output = self.lm_head(zHs)[:, self.puzzle_emb_len:]
        diff_to_hs = self.transcoder(zH_next)
        q_logits = self.q_head(zH_next[:, 0]).to(torch.float32)

        return diff_to_hs.squeeze(1), zL_next, zH_next
