from dataclasses import dataclass


@dataclass
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

    # replacement_method: str = "-1" # or 0.5, or ie+supressed[0.5:] or hs+supressed[0.5:] or supressed[0.5:]
    # replacement_method: str = "0.5"
    # replacement_method: str = "supressed[0.5:]"
    # replacement_method: str = "hs+supressed[0.5:]"
    replacement_method: str = "ie+supressed[0.5:]"
    
  
    # use bf16 for all model weights: should work but seems to fail ??
    bf16: bool = True

    # use bf16 for linear and conv weights (not emb or norm): highly experimental
    bf16_weight: bool = False # experimental, doesn't seem to work

    # use 8 bit opt gradients): experimental
    opt_8b: bool = False
    
    cot_epochs: int = 2 # 6-10 in paper but those were full length
    epochs_per_stage: int = 5 # how many epochs to train for each stage
    max_latent_stage: int = 3 # max number of thought tokens, unstable is >3?
    num_epochs: int = 50 # 50 in coconut, but more capable model probobly need less

    # effectve batch size should be 128
    batch_size_training: int = 12
    gradient_accumulation_steps: int = 10

    # https://github.com/QwenLM/Qwen3/blob/714df5bce80a67c698e37034e71dc2efa19ceaf3/examples/llama-factory/qwen2-7b-full-sft.yaml#L27
    lr: float = 1e-4 # 1e-4 in coconut, but 1e-6 in verl
    weight_decay: float = 0.01 # 0.01 in coconut, 0 in verl
    grad_clip: float = 10.0
    scheduler: str = "cosine" # "constant" or "cosine" or "linear"

    debug: bool = False

    seed: int = 42

    reset_optimizer: bool = True

    loss_seq_vcr: bool = False # experimental loss, might help with intermediate state stabiliy



    # # used to get a baseline, not used or broken now?
    # no_cot: bool = False
    # no_thoughts: bool = False
    # coconut: bool = True
    # cot: bool = False

    max_size: int = 60000 # full ~400k in coconut
    
    # dataset: for each reasoning step we use X thought tokens (up to our max)
    c_thought: int = 2

    # dataset
    pad_latent_to_max: bool = True

    # dataset: with some prob, randomly sample earlier stage
    uniform_prob: float = 0.0

    train_path: str = "data/gsm_train.json"
    val_path: str = "data/gsm_valid.json"

    # this seems like it should help, but it seems to make it worse
    system_prompt: str = ""
    # system_prompt: str = "Solve this math question with multiple steps like `<<5*0+1*2=?>>` OR silently within `<|start-latent|><|end-latent|>`. Then return the final answer like `### 2\n`. Save all comments until after the answer."

@dataclass
class GSMQwen(BaseConfig):
    train_path: str = "data/gsm_train.json"
    val_path: str = "data/gsm_valid.json"


class GSMQwenResume(BaseConfig):
    model_id:str = "outputs/qwen3-0.6b_20250514-194730/checkpoint_2"

class Debug(GSMQwen):
    model_id: str = "yujiepan/qwen3-tiny-random"
    resume_epochs: int = 2
    cot_epochs: int = 2
    loss_seq_vcr: bool = True

@dataclass
class GsmQwen_H100(GSMQwen):
    """
    For running on a h100
    """
    name: str = "gsm-qwen-1.5b"
    
    bf16: bool = True
    bf16_weight: bool = False
    opt_8b: bool = False
    batch_size_training: int = 32
    gradient_accumulation_steps: int = 4
    reset_optimizer: bool = True
