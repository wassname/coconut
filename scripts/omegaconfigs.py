from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class BaseConfig:
    project: str = "coconut"
    save_path: str = "outputs/"
    name: str = "qwen-1.5b"
    
    only_eval: bool = False
    
    coconut: bool = True
    cot: bool = False
    no_thoughts: bool = False
    no_cot: bool = False
    
    c_thought: int = 2
    epochs_per_stage: int = 2
    max_latent_stage: int = 4
    pad_latent_to_max: bool = True
    # replacement_method: str = "-1" # or 0.5, or ie+supressed[0.5:] or hs+supressed[0.5:] or supressed[0.5:]
    # replacement_method: str = "0.5"
    replacement_method: str = "supressed[0.5:]"
    # replacement_method: str = "hs+supressed[0.5:]"
    # replacement_method: str = "ie+supressed[0.5:]"
    
    
    uniform_prob: float = 0.0
    model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    
    load_model_path: str = "" # set to checkpoint
    resume: int = 0 # set to phase/epoch

    seed: int = 0
    bf16: bool = True
    bf16_weight: bool = False # experimental
    opt_8b: bool = False
    
    train_path: str = "data/gsm_train.json"
    val_path: str = "data/gsm_valid.json"
    batch_size_training: int = 26
    max_size: int = 14000 # full ~40k in coconut
    debug: bool = False
    gradient_accumulation_steps: int = 2
    num_epochs: int = 25 # 50 in coconut
    lr: float = 1e-4 # 1e-4 in coconut, but 1e-6 in verl
    weight_decay: float = 0.0 # 0.01 in coconut, 0 in verl


@dataclass
class GSMQwenConfig(BaseConfig):
    name: str = "gsm-qwen-1.5b"
    model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    train_path: str = "data/gsm_train.json"
    val_path: str = "data/gsm_valid.json"

@dataclass
class GsmQwen1_5b_H100(GSMQwenConfig):
    name: str = "gsm-qwen-1.5b"
    replacement_method: str = "-2"
    
    load_model_path: str = ""
    resume: int = 0
    
    bf16: bool = True
    bf16_weight: bool = False
    opt_8b: bool = True
    batch_size_training: int = 30
    gradient_accumulation_steps: int = 1

# # ...existing code...
# if __name__ == "__main__":
#     cfg = OmegaConf.structured(GsmQwen1_5b_H100)
#     print(OmegaConf.to_yaml(cfg))
