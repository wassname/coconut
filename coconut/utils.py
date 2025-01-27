# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary

    def __getitem__(self, key):
        return self.__dict__[key]


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



from transformers.trainer_callback import ProgressCallback
class ProgressCallbackNoPrint(ProgressCallback):
    """ProgressCallback that doesn't print anything
    
    Usage: 
        # at top of file, monkey patch the default progress callback
        import transformers
        transformers.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNoPrint

        # or 

        trainer = Trainer(
            ..., 
            logging_steps=1,
            callbacks=[ProgressCallbackNoPrint()]
        )
        rm_old_prog_cb(trainer)
        trainer.train()
    """
    def on_log(self, *args, **kwargs):
        pass

def rm_old_prog_cb(trainer):
    for cb in trainer.callback_handler.callbacks:
        if isinstance(cb, ProgressCallback):
            if not isinstance(cb, ProgressCallbackNoPrint):
                trainer.remove_callback(cb)
