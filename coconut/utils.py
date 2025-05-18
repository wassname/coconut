# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np
import gc
import torch.nn as nn
from argparse import Namespace
from loguru import logger


class Config(Namespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

# class Config:
#     # to access a dict with object.key
#     def __init__(self, dictionary):
#         self.__dict__ = dictionary

#     def __getitem__(self, key):
#         return self.__dict__[key]
    

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_bfloat16(module, verbose=False):
    for child in module.children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            child.to(torch.bfloat16)
            if verbose:
                print(f"Converted {child} to bfloat16")
        else:
            convert_to_bfloat16(child)


def print_cuda_devices():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.get_device_name(i)
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(torch.cuda.get_device_capability(i))
            logger.info(torch.cuda.get_device_properties(i))
    else:
        logger.warning("CUDA is not available")


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
