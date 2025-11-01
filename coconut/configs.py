# from dataclasses import dataclass
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Optional, List, Literal, ClassVar

from coconut.trmlora.recursive_lora import TRMLoraAConfig
from coconut.trmlora.recursive_delora import TRMDeloraAConfig
from coconut.trmlora.recursive_hra import TRMHraAConfig
from coconut.trmlora.recursive_svft import TRMSvftAConfig

# latent_token = "<|latent|>"
# bot_token = "<|start-latent|>"
# eot_token = "<|end-latent|>"
# TODO it would be nicer to search for a token us tokenizer.encode(" ...")[0], so we can get whitepace right
# bot_token = "Hmm"
# latent_token = "..."
# eot_token = "Therefore"

@dataclass(config=ConfigDict(validate_assignment=True)) 
class BaseConfig:
    """Base COCONUT config: full model training with latent reasoning."""
    project: str = "coconut"
    save_path: str = "outputs/"
    name: str = "qwen3-0.6b"
    model_id: str = "suayptalha/Qwen3-0.6B-Math-Expert"
    
    only_eval: bool = False
    
    load_model_path: str = ""  # set to checkpoint path to resume
    resume_epochs: int = 1  # set to phase/epoch to resume from
    
    # replacement_method: str = "supressed[0.75:]"  # how to replace latent tokens: 0.5, -3, supressed[0.5:], hs+supressed[0.5:], ie+supressed[0.5:]
    use_position_ids: bool = True  # experimental, might help model mode switch to latent tokens
    
    bf16: bool = True  # use bf16 for all model weights
    bf16_weight: bool = False  # use bf16 for linear and conv weights only (not emb or norm) - experimental
    opt_8b: bool = False  # use 8 bit optimizer gradients - experimental
    load_in_4bit: bool = False  # load LLM in 4bit (for TRM mode)
    load_in_8bit: bool = False  # load LLM in 8bit (for TRM mode)
    
    cot_epochs: int = 2  # 6-10 in paper but those were full length
    epochs_per_stage: int = 8  # how many epochs to train for each stage
    max_latent_stage: int = 3  # max number of thought tokens, unstable if >3
    num_epochs: int = 50  # 50 in coconut, but more capable model probably needs less
    
    batch_size_training: int = 12  # effective batch size should be 128
    gradient_accumulation_steps: int = 10
    
    lr: float = 1e-4  # 1e-4 in coconut, 1e-6 in verl
    weight_decay: float = 0.1  # 0.01 in coconut, 0 in verl, 0.1-1 in TRM paper
    grad_clip: float = 10.0
    scheduler: str = "cosine"  # "constant" or "cosine" or "linear"
    
    debug: bool = False
    seed: int = 42
    reset_optimizer: bool = False
    
    loss_seq_vcr: bool = False  # experimental loss, might help with intermediate state stability
    # n_detached_recursions: int = 2  # 0=disabled, 2=keep gradients only for last 2 passes (TRM-style)
    
    collect_hs: bool = False  # whether to collect hidden states during forward pass
    
    max_size: int = 60_000  # full dataset ~400k in coconut
    c_thought: int = 1  # latent tokens per reasoning step (stage N → N * c_thought tokens)
    pad_latent_to_max: bool = True
    uniform_prob: float = 0.0  # with some prob, randomly sample earlier stage
    
    train_path: str = "data/gsm_train.json"
    val_path: str = "data/gsm_valid.json"
    
    system_prompt: str = ""  # seems to make it worse when set
    
    latent_token_id: Optional[int] = None  # to be set when loading model
    bot_token_id: Optional[int] = None  # beginning of thought token id
    eot_token_id: Optional[int] = None  # end of thought token id
    eos_token_id: Optional[int] = None  # for generate pad/eos

    latent_token: str = '...' # latent reasoning token, we chose something that is in the tokenizer vocab, that makes a phrase not to out of sample for training, and is a single token. e.g. ["Hmm", "...", "...", "Therefore"]
    bot_token: str = "Hmm"
    eot_token: str = 'Therefore'


@dataclass
class TRMConfig(BaseConfig):
    """TRM base config: shared settings for all TRM adapter modes."""
    # see COCOCNUT https://github.com/facebookresearch/coconut/blob/27273cb8cca4bb763c041a63b036d0c3b7cbbb48/args/gsm_coconut.yaml#L34
    # see TRM https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9/config/arch/trm.yaml paper
    resume_epochs: int = 1
    cot_epochs: int = 0
    skip_stage_zero: bool = True  # skip stage 0 : <start_latent><end_latent> training with 0 latent tokens
    num_epochs: int = 6
    epochs_per_stage: int = 3
    
    scheduler: str = "linear"
    lr: float = 4e-3 # 1e-4 in paper
    weight_decay: float = 1. # 1 and 0.1 in TRM paper. 0.01 in COCONUT paper. 
    
    max_size: int = 10_000
    batch_size_training: int = 12
    gradient_accumulation_steps: int = 6 # 768 // 14 # paper had effective batch size of 768

    eval_first_epoch: bool = False
    loss_nll_ratio_margin: bool = False
    
    trm_h_cycles: int = 3  # high level recursive cycles (T=3 in repo)
    trm_l_cycles: int = 6  # low level recursive cycles (n=6 in repo)
    trm_l_layers: int = 2  # layers for L_net, 2 best in paper/repo
    trm_num_heads: int = 4  # number of attention heads in TRM, 8 in repo
    trm_expansion: float = 2  # MLP expansion factor in TRM, 4 in repo, meaning it expands to 4*512=2048. But we are expanding from a lower rank so might want hs/rank=2048/18=114

    trm_persistent_steering: bool = True  # persistent steering vector across recursions

    layers_spacing_adapter: int = 5  # number of spaced out layers to apply adapter to, larger number means all
    layers_start_adapter: float = 0.3  # start layer fraction to apply adapter
    layers_end_adapter: float = 0.95  # end layer fraction to apply adapter

    target_modules_pattern: Optional[str] = None   # regex pattern to match target module names, best performance for '.+\.(gate_proj).*$' 


@dataclass
class TRMLoRA(TRMConfig):
    """TRM LoRA mode: inline recursive LoRA adapter on frozen LLM."""
    _adapter_class: ClassVar = TRMLoraAConfig
    name: str = "trmlora-qwen3-0.6b"
    use_trm_lora: bool = True

    adapter_r: int = 8  # LoRA rank
    adapter_lora_alpha: int = 32  # LoRA alpha scaling
    # adapter_dropout: float = 0.0


@dataclass
class TRMDelora(TRMConfig):
    """TRM DeLoRA mode: inline recursive DeLoRA adapter on frozen LLM.
    https://openreview.net/forum?id=X1U74IwuxG
    """
    _adapter_class: ClassVar = TRMDeloraAConfig
    name: str = "trmdelora-qwen3-0.6b"
    use_trm_delora: bool = True
    
    adapter_r: int = 8  # DeLoRA rank
    adapter_delora_lambda: int = 30  # DeLoRA lambda
    # adapter_dropout: float = 0.0
    # lr=1e-3 # delora paper shows it supports a higher lr
    


@dataclass
class TRMHra(TRMConfig):
    """TRM HRA mode: inline recursive HRA adapter on frozen LLM."""
    _adapter_class: ClassVar = TRMHraAConfig
    name: str = "trmhra-qwen3-0.6b"
    use_trm_hra: bool = True

    adapter_r: int = 8  # HRA rank (even recommended for symmetric init)
    adapter_hra_alpha: int = 16  # Scaling for TRM refinement delta
    # adapter_dropout: float = 0.0

    adapter_hra_apply_GS: bool = False  # Gram-Schmidt orthogonalization


@dataclass
class TRMSvft(TRMConfig):
    """TRM SVFT mode: inline recursive SVFT adapter on frozen LLM.
    
    Hybrid SVD: principal_rank for top SVD (base variance) + tail_rank for low-rank approx of remaining vectors (subtle info).
    Hypothesis: Merges tail without full-rank memory; principal leverages pretrain, tail recovers truncation loss—better than pure top-k or random LoRA.
    """
    _adapter_class: ClassVar = TRMSvftAConfig
    name: str = "trmsvft-qwen3-0.6b"
    use_trm_svft: bool = True

    adapter_r: int = 42  # Total rank (principal + tail; low for memory)
    adapter_fill_orthonormal: bool = False  # Disable for hybrid (principal + tail covers)
    # adapter_principal_rank: int = 32  # Top SVD for principal directions
    adapter_tail_rank: int = 12  # Low-rank approx for tail merging
    adapter_svft_mode: Literal["replace_add", "replace_mul", "adapter_add", "adapter_mult"] = "adapter_add"

@dataclass
class Debug:
    """Fast iteration TRM LoRA config with tiny model."""
    model_id: str = "yujiepan/qwen3-tiny-random"
    name: str = "trm-debug"
    
    max_size: int = 1000
    debug: bool = True
    
    batch_size_training: int = 8
    gradient_accumulation_steps: int = 2
    
    cot_epochs: int = 1
    epochs_per_stage: int = 1
    resume_epochs: int = 0
    num_epochs: int = 5
    
    eval_first_epoch: bool = False


@dataclass
class TRMLoRADebug(Debug, TRMLoRA):
    """Fast iteration TRM LoRA config with tiny model."""
    name: str = "trmlora-debug"

@dataclass
class TRMHraDebug(Debug, TRMHra):
    """Fast iteration TRM HRA config with tiny model."""
    name: str = "trmhra-debug"

@dataclass
class TRMDeloraDebug(Debug, TRMDelora):
    """Fast iteration TRM DeLoRA config with tiny model."""
    name: str = "trmdelora-debug"

@dataclass
class TRMSvftDebug(Debug, TRMSvft):
    """Fast iteration TRM SVFT config with tiny model."""
    name: str = "trmsvft-debug"
