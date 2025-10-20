# from dataclasses import dataclass
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import Optional, List, Literal


@dataclass(config=ConfigDict(validate_assignment=True)) 
class BaseConfig:
    project: str = "coconut"
    save_path: str = "outputs/"
    name: str = "qwen3-0.6b"
    # model_id: str = "Qwen/Qwen3-0.6B"
    model_id: str = "suayptalha/Qwen3-0.6B-Math-Expert"
    
    
    
    only_eval: bool = False
    
    # to resume set the two below
    load_model_path: str = "" # set to checkpoint
    resume_epochs: int = 0 # set to phase/epoch

    replacement_method: str = "supressed[0.75:]" # or 0.5, or ie+supressed[0.5:] or hs+supressed[0.5:] or supressed[0.5:]
    # replacement_method: str = "0.5"
    # replacement_method: str = "-3"
    # replacement_method: str = "supressed[0.5:]"
    # replacement_method: str = "hs+supressed[0.5:]"
    # replacement_method: str = "ie+supressed[0.5:]"
    # ETHER hs[-2]

    use_position_ids: bool = True # experimental, might help the model mode switch to latent tokens
    
  
    # use bf16 for all model weights: should work but seems to fail ??
    bf16: bool = True

    # use bf16 for linear and conv weights (not emb or norm): highly experimental
    bf16_weight: bool = False # experimental, doesn't seem to work

    # use 8 bit opt gradients): experimental
    opt_8b: bool = False
    
    cot_epochs: int = 2 # 6-10 in paper but those were full length
    epochs_per_stage: int = 8 # how many epochs to train for each stage
    max_latent_stage: int = 3 # max number of thought tokens, unstable is >3?
    num_epochs: int = 50 # 50 in coconut, but more capable model probobly need less

    # effectve batch size should be 128
    batch_size_training: int = 12
    gradient_accumulation_steps: int = 10

    # https://github.com/QwenLM/Qwen3/blob/714df5bce80a67c698e37034e71dc2efa19ceaf3/examples/llama-factory/qwen2-7b-full-sft.yaml#L27
    lr: float = 1e-4 # 1e-4 in coconut, but 1e-6 in verl
    weight_decay: float = 0.1 # 0.01 in coconut, 0 in verl # 0.1 anmd 1 in TRM paper
    grad_clip: float = 10.0
    scheduler: str = "linear" # "constant" or "cosine" or "linear"

    debug: bool = False

    seed: int = 42

    reset_optimizer: bool = False

    loss_seq_vcr: bool = False # experimental loss, might help with intermediate state stabiliy

    # TRM-style: detach gradients for first N passes, only backprop through last few
    n_detached_recursions: int = 2  # 0=disabled, 2=keep gradients only for last 2 passes

    # TRM mode: use frozen quantized LLM with TRM recursive reasoning
    use_trm: bool = False  # Enable TRM adapter with frozen LLM
    load_in_4bit: bool = True  # Load LLM in 4bit for TRM mode
    load_in_8bit: bool = False  # Load LLM in 8bit for TRM mode



    # # used to get a baseline, not used or broken now?
    # no_cot: bool = False
    # no_thoughts: bool = False
    # coconut: bool = True
    # cot: bool = False

    max_size: int = 60_000 # full ~400k in coconut
    
    # dataset: for each reasoning step we use X thought tokens (up to our max)
    c_thought: int = 1

    # dataset
    pad_latent_to_max: bool = True

    # dataset: with some prob, randomly sample earlier stage
    uniform_prob: float = 0.0

    train_path: str = "data/gsm_train.json"
    val_path: str = "data/gsm_valid.json"

    # this seems like it should help, but it seems to make it worse
    system_prompt: str = ""
    # system_prompt: str = "Solve this math question with multiple steps like `<<5*0+1*2=?>>` OR silently within `<|start-latent|><|end-latent|>`. Then return the final answer like `### 2\n`. Save all comments until after the answer."

    latent_token_id: Optional[int] = None  # to be set when loading model

# @dataclass
# class GSMQwen(BaseConfig):
#     train_path: str = "data/gsm_train.json"
#     val_path: str = "data/gsm_valid.json"

# @dataclass
# class GSMQwenResume(BaseConfig):
#     load_model_path: str = "outputs/qwen3-0.6b_20250514-194730/checkpoint_2"
#     cot_epochs: int = 2
#     epochs_per_stage: int = 8
#     resume_epochs: int = 1
#     # loss_seq_vcr: bool = True



# @dataclass
# class GsmQwen_H100(GSMQwenResume):
#     """
#     For running on a h100
#     """
#     name: str = "gsm-qwen-0.6bh100"
    
#     bf16: bool = True
#     bf16_weight: bool = False
#     opt_8b: bool = False

#     # note 48 ran out of ram at stage 3
#     batch_size_training: int = 48
#     gradient_accumulation_steps: int = 3
#     max_size: int = 60_000 # full ~400k in coconut


# @dataclass
# class EpochSingleCoT(GsmQwen_H100):
#     load_model_path: str = "outputs/qwen3-0.6b_20250514-194730/checkpoint_2"
#     cot_epochs: int = 2
#     epochs_per_stage: int = 1
#     resume_epochs: int = 1
#     # loss_seq_vcr: bool = True
#     num_epochs: int = 2
#     max_size: int = 8_000 # full ~400k in coconut

# @dataclass
# class EpochSingleLatent(GsmQwen_H100):
#     load_model_path: str = "outputs/qwen3-0.6b_20250514-194730/checkpoint_2"
#     cot_epochs: int = 2
#     epochs_per_stage: int = 1
#     resume_epochs: int = 3
#     # loss_seq_vcr: bool = True
#     num_epochs: int = 4
#     max_size: int = 8_000 # full ~400k in coconut

@dataclass
class Debug(BaseConfig):
    model_id: str = "yujiepan/qwen3-tiny-random"

    max_size: int = 1000 # full ~400k in coconut
    debug: bool = True

    batch_size_training: int = 8
    gradient_accumulation_steps: int = 2

    cot_epochs: int = 1
    epochs_per_stage: int = 1
    resume_epochs: int = 3


@dataclass
class TRMTest(BaseConfig):
    """
    TRM-style detached recursions test.
    
    Detaches gradients for early recursive passes, only backprops through last N.
    Forces model to learn to clean up its own accumulated errors.
    """
    # TRM experiment: only backprop last 2 passes out of 4 total
    n_detached_recursions: int = 2

@dataclass
class TRM(BaseConfig):
    """Full TRM mode: frozen quantized LLM + TRM adapter."""
    name: str = "trm-qwen3-0.6b"
    load_in_4bit: bool = True

    # load_model_path: str = "outputs/qwen3-0.6b_20250514-194730/checkpoint_2"

    # resume one that already know CoT reasoning
    model_id: str = "outputs/qwen3-0.6b_20250514-194730/checkpoint_2"

    # num_epochs: int = 6  # Just a few epochs to test training
    resume_epochs: int = 8
    # epochs_per_stage: int = 8
    cot_epochs: int = 0
    num_epochs: int = 30  # More epochs to let TRM adapter learn
    lr: float = 1e-4  # Slightly lower LR for adapter training

    eval_first_epoch: bool = True  # Evaluate before training

    # NOTE: see TRM paper settings https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9/config/arch/trm.yaml#L17
    use_trm: bool = True

    trm_n_sup: int = 16  # Deep supervision steps (N_sup in HRM paper)
    # trm_h_layers: int = 0  # Layers for H_net (0 for single net mode)
    trm_h_cycles: int = 3  # Outer cycles (T in paper)
    trm_l_cycles: int = 6  # Inner cycles (n in paper)
    trm_l_layers: int = 2  # Layers for L_net (or single net)
    trm_hidden_size: Optional[int] = None  # Dynamic from base model if None
    trm_num_heads: int = 8  # Number of attention heads in TRM (12 in paper)
    trm_expansion: float = 2.67  # MLP expansion factor in TRM (4 in paper)
    trm_transcoder_layers: int = 3  # Number of SwiGLU layers in transcoder (configurable)

    trm_svd_init: bool = True  # Whether to use SVD-based initialization for TRM transcoder

    trm_persistent_steering: bool = True  # Whether to persistently steer all future latent embeddings

    # n_detached_recursions: int = 2  # Number of detached recursions (paper used >6)
    # n_gradient_recursions: int = 2  # Number of final recursions with gradients (paper uses 2)

    max_size: int = 20_000  # Start very small for testing
    batch_size_training: int = 16  # Reduced from 16 due to OOM
    gradient_accumulation_steps: int = 128//16  # Keep effective batch ~128

    eval_first_epoch: bool = False  # Skip eval for speed
    

@dataclass
class TRMDebug(TRM):
    model_id: str = "yujiepan/qwen3-tiny-random"

    max_size: int = 1000 # full ~400k in coconut
    debug: bool = True

    batch_size_training: int = 8
    gradient_accumulation_steps: int = 2

    cot_epochs: int = 1
    epochs_per_stage: int = 1
    resume_epochs: int = 3
    num_epochs: int = 5

    eval_first_epoch: bool = False  # Skip eval for speed

# @dataclass
# class TRM_H100(GsmQwen_H100):
#     """
#     TRM-style detached recursions test.
    
#     Detaches gradients for early recursive passes, only backprops through last N.
#     Forces model to learn to clean up its own accumulated errors.
#     """
#     # TRM experiment: only backprop last 2 passes out of up to 8 total
#     n_detached_recursions: int = 2
