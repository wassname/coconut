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
    grad_clip: float = 1.0
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


    loss_seq_vcr: bool = True
    collect_hs: bool = False  # whether to collect hidden states during forward pass



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
    bot_token_id: Optional[int] = None  # beginning of thought token id
    eot_token_id: Optional[int] = None  # end of thought token id
    eos_token_id: Optional[int] = None  # for generate pad/eos

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
class TRMLoRA(BaseConfig):
    """
    TRM LoRA mode: inline recursive LoRA adapter on frozen LLM.
    """
    name: str = "trmlora-qwen3-0.6b"
    load_in_4bit: bool = False
    model_id = "suayptalha/Qwen3-0.6B-Math-Expert"
    resume_epochs: int = 8
    cot_epochs: int = 0
    num_epochs: int = 25
    lr: float = 4e-3
    use_trm: bool = False
    trm_h_cycles: int = 2
    trm_l_cycles: int = 2
    trm_l_layers: int = 2
    trm_hidden_size: Optional[int] = None
    trm_num_heads: int = 2
    trm_expansion: float = 2.67
    trm_transcoder_layers: int = 1
    loss_reg_ie_diff: bool = True
    loss_nll_ratio_margin: bool = True
    max_size: int = 20_000
    batch_size_training: int = 16
    gradient_accumulation_steps: int = 6
    # TRMLoRA-specific fields
    use_trm_lora: bool = True
    epochs_per_stage: int = 6
    lora_r: int = 12
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_layers : int = 4
    eval_first_epoch: bool = False
    debug: bool = False
    weight_decay: float = 0.0


@dataclass
class TRMLoRADebug(TRMLoRA):
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
