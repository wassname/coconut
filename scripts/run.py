import coconut.silence
import coconut.trmlora  # MUST be first to register PEFT methods before any peft imports

from torchinfo import summary
import os
import time
from copy import copy
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import torch
import torch.optim as optim
import yaml
from loguru import logger
from torch import nn
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)


import wandb

from coconut.dataset import (
    get_cot_latent_dataset,
    get_dataset,
    get_question_only_latent_dataset,
)
from coconut.eval import evaluate, get_answer_perplexity, get_answer_preference
from coconut.utils import Config, convert_to_bfloat16, set_seed, clear_memory, print_cuda_devices
from coconut.load_model import (
    load_new_model,
    resume_model,
    save_model,
)
from coconut.train import train

logger.remove()


def sink(msg):
    return tqdm.write(msg, end="")


logger.add(sink, colorize=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"



def main():
    import coconut.silence
    import tyro
    from coconut import configs # this will be my dataclass files
    # experiments = configs.__dict__.keys()
    # logger.debug(f"Available experiments: {experiments}")
    # parser = argparse.ArgumentParser(description="coconut")
    # parser.add_argument("experiment", type=str, help=f"experiment names: [{experiments}]")
    # args = parser.parse_args()

    # logger.debug(f"Command line arguments: {os.sys.argv}")
    ConfigCls = getattr(configs, os.sys.argv[1])
    conf = tyro.cli(ConfigCls, args=os.sys.argv[2:], use_underscores=True)
    return train(conf)

if __name__ == "__main__":
    # import coconut.silence
    main()
