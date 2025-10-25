# from dataclasses import dataclass
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Optional, List, Literal


@dataclass(config=ConfigDict(validate_assignment=True)) 
class BaseConfig:
    """Base COCONUT config: full model training with latent reasoning."""
    project: str = "coconut"
    save_path: str = "outputs/"
    name: str = "qwen3-0.6b"
    model_id: str = "suayptalha/Qwen3-0.6B-Math-Expert"
    
    only_eval: bool = False
    
    load_model_path: str = ""  # set to checkpoint path to resume
    resume_epochs: int = 0  # set to phase/epoch to resume from
    
    replacement_method: str = "supressed[0.75:]"  # how to replace latent tokens: 0.5, -3, supressed[0.5:], hs+supressed[0.5:], ie+supressed[0.5:]
    use_position_ids: bool = True  # experimental, might help model mode switch to latent tokens
    
    bf16: bool = True  # use bf16 for all model weights
    bf16_weight: bool = False  # use bf16 for linear and conv weights only (not emb or norm) - experimental
    opt_8b: bool = False  # use 8 bit optimizer gradients - experimental
    
    cot_epochs: int = 2  # 6-10 in paper but those were full length
    epochs_per_stage: int = 8  # how many epochs to train for each stage
    max_latent_stage: int = 3  # max number of thought tokens, unstable if >3
    num_epochs: int = 50  # 50 in coconut, but more capable model probably needs less
    
    batch_size_training: int = 12  # effective batch size should be 128
    gradient_accumulation_steps: int = 10
    
    lr: float = 1e-4  # 1e-4 in coconut, 1e-6 in verl
    weight_decay: float = 0.1  # 0.01 in coconut, 0 in verl, 0.1-1 in TRM paper
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # "constant" or "cosine" or "linear"
    
    debug: bool = False
    seed: int = 42
    reset_optimizer: bool = False
    
    loss_seq_vcr: bool = False  # experimental loss, might help with intermediate state stability
    n_detached_recursions: int = 2  # 0=disabled, 2=keep gradients only for last 2 passes (TRM-style)
    
    load_in_4bit: bool = False  # load LLM in 4bit (for TRM mode)
    load_in_8bit: bool = False  # load LLM in 8bit (for TRM mode)
    collect_hs: bool = False  # whether to collect hidden states during forward pass
    
    max_size: int = 60_000  # full dataset ~400k in coconut
    c_thought: int = 1  # for each reasoning step we use X thought tokens (up to our max)
    pad_latent_to_max: bool = True
    uniform_prob: float = 0.0  # with some prob, randomly sample earlier stage
    
    train_path: str = "data/gsm_train.json"
    val_path: str = "data/gsm_valid.json"
    
    system_prompt: str = ""  # seems to make it worse when set
    
    latent_token_id: Optional[int] = None  # to be set when loading model
    bot_token_id: Optional[int] = None  # beginning of thought token id
    eot_token_id: Optional[int] = None  # end of thought token id
    eos_token_id: Optional[int] = None  # for generate pad/eos

@dataclass
class TRMLoRA(BaseConfig):
    """TRM LoRA mode: inline recursive LoRA adapter on frozen LLM."""
    name: str = "trmlora-qwen3-0.6b"
    model_id: str = "suayptalha/Qwen3-0.6B-Math-Expert"
    
    load_in_4bit: bool = False
    use_trm_lora: bool = True
    
    resume_epochs: int = 2
    cot_epochs: int = 0
    num_epochs: int = 20
    epochs_per_stage: int = 10
    
    lr: float = 1e-3
    weight_decay: float = 0.01
    
    max_size: int = 20_000
    batch_size_training: int = 10
    gradient_accumulation_steps: int = 700 // 10

    eval_first_epoch: bool = False
    loss_nll_ratio_margin: bool = False
    
    lora_r: int = 18  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha scaling
    lora_dropout: float = 0.0
    lora_layers: int = 8  # number of spaced out layers to apply LoRA to
    trm_h_cycles: int = 3  # high level recursive cycles (T in paper)
    trm_l_cycles: int = 2  # low level recursive cycles (n in paper)
    trm_l_layers: int = 2  # layers for L_net
    trm_num_heads: int = 3  # number of attention heads in TRM
    trm_expansion: float = 4  # MLP expansion factor in TRM




@dataclass
class TRMDelora(TRMLoRA):
    """TRM DeLoRA mode: inline recursive DeLoRA adapter on frozen LLM."""
    name: str = "trmdelora-qwen3-0.6b"
    use_trm_delora: bool = True
    use_trm_lora: bool = False
    
    delora_r: int = 18  # DeLoRA rank
    delora_lambda: int = 15  # DeLoRA lambda
    delora_dropout: float = 0.0
    delora_layers: int = 8  # number of spaced out layers to apply DeLoRA to
    trm_h_cycles: int = 3  # high level recursive cycles (T in paper)
    trm_l_cycles: int = 2  # low level recursive cycles (n in paper)
    trm_l_layers: int = 2  # layers for L_net
    trm_num_heads: int = 3  # number of attention heads in TRM
    trm_expansion: float = 4  # MLP expansion factor in TRM




@dataclass
class TRMLoRADebug(TRMLoRA):
    """Fast iteration TRM LoRA config with tiny model."""
    model_id: str = "yujiepan/qwen3-tiny-random"
    name: str = "trmlora-debug"
    
    max_size: int = 1000
    debug: bool = True
    
    batch_size_training: int = 8
    gradient_accumulation_steps: int = 2
    
    cot_epochs: int = 1
    epochs_per_stage: int = 1
    resume_epochs: int = 3
    num_epochs: int = 5
    
    eval_first_epoch: bool = False


@dataclass
class TRMHra(TRMLoRA):
    """TRM HRA mode: inline recursive HRA adapter on frozen LLM."""
    name: str = "trmhra-qwen3-0.6b"
    use_trm_hra: bool = True
    use_trm_delora: bool = False
    use_trm_lora: bool = False
    
    hra_r: int = 8  # HRA rank (even recommended for symmetric init)
    hra_apply_GS: bool = False  # Gram-Schmidt orthogonalization
    hra_alpha: int = 16  # Scaling for TRM refinement delta
    hra_dropout: float = 0.0
    hra_layers: int = 8  # number of spaced out layers to apply HRA to
    trm_h_cycles: int = 3  # high level recursive cycles (T in paper)
    trm_l_cycles: int = 2  # low level recursive cycles (n in paper)
    trm_l_layers: int = 2  # layers for L_net
    trm_num_heads: int = 3  # number of attention heads in TRM
    trm_expansion: float = 4  # MLP expansion factor in TRM