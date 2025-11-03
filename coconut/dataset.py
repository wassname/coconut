# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
from loguru import logger
import itertools
import random
from dataclasses import dataclass
from typing import Optional
import multiprocessing as mp
import os
import re
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from anycache import anycache

DEBUG = os.environ.get("DEBUG", False)

default_num_proc = None if DEBUG else mp.cpu_count()//2 

@anycache('.anycache')
def get_dataset(path, tokenizer, max_size=1000000000, drop_unused=True, system_prompt="", num_proc=default_num_proc, verbose=True):
    if system_prompt:
        # system_prompt = "<|im_start|>system\n" + system_prompt.strip() + "<|im_end|>\n"
        system_prompt = system_prompt.strip() + "\n"

    # pre_q = "<|im_start|>user\n"
    pre_q = ""

    # encode = tokenizer.encode
    # encode = tokenizer.apply_chat_template

    # post_q = "<|im_end|>\n<|im_start|>assistant\n"
    post_q = ""

    def tokenize_sample(sample):
        """We want parts of the quesiton tokenized separately, but we also want to use the chat template. Easiest way is chat_template to text -> split -> encode"""


        sep = "<|split|>"

        steps = f"\n{sep}".join(sample["steps"])+"\n"
        ans = f"{sep}### {sample['answer']}\n"

        messages_split = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pre_q + sample["question"] + post_q},
            {"role": "assistant", "content": sep+steps+ans},
        ]
        text = tokenizer.apply_chat_template(messages_split, tokenize=False)
        parts = text.split(sep)
        parts_t = [tokenizer.encode(p, add_special_tokens=False) for p in parts]

        n_steps = len(sample["steps"])
        assert len(parts_t) == n_steps + 2, f"{len(parts_t)} != {n_steps+2} {text} {sample['steps']}"
        question_tokenized = parts_t[0]
        steps_tokenized = parts_t[1:n_steps+1]

        # FIXME this is '### 300\n<|im_end|>\n', not just the answer
        # ['###', ' ', '3', '0', '0', '\n', '<|im_end|>', '\n']
        answer_tokenized = parts_t[n_steps+1]
        rexp = re.compile(r'\d+\.?\d*')
        answer_mask = [rexp.match(x) is not None for x in tokenizer.batch_decode(answer_tokenized)]

        sample = {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "answer_mask": answer_mask,
            "idx": sample["idx"],
            # "messages_split": messages_split,
            # "text": text,
        }
        return sample

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    dataset_tok = dataset.map(
        tokenize_sample, remove_columns=list(dataset.features) if drop_unused else None, num_proc=num_proc, 
        desc=f'tokenize_sample: {path}'
    )

    row = dataset_tok[0]
    if verbose:
        logger.debug("Example row:")
        logger.debug("question_tokenized", tokenizer.decode(row["question_tokenized"]))
        for i, s in enumerate(row["steps_tokenized"]):
            logger.debug(f"steps_tokenized[{i}] {tokenizer.decode(s).strip()}")
        logger.debug(f'answer_tokenized {tokenizer.decode(row["answer_tokenized"])}', )

    return dataset_tok

@anycache('.anycache')
def get_question_only_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    bot_id,
    latent_id,
    eot_id,
    no_bot_eot=False,
    skip_bot_eot=True,
    drop_unused=True,
    num_proc=default_num_proc,
):
    """    
    for testing, with no reasoning or ans

    format: question, latent

    args:
    - no_bot_eot: if True, don't include thought tokens 
    """

    no_bot_eot = scheduled_stage < 0 or skip_bot_eot

    def process_dataset(sample):

        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            # max on thought per reasoning step
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        # we increase the amount of thought steps as we progress throught the coconut epochs
        k = min(max_latent_stage, scheduled_stage)
        k = max(0, k)

        
        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_bot_eot else [bot_id])
            + [latent_id] * k
            + ([] if no_bot_eot else [eot_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            # "position_ids": list(range(len(tokens))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features) if drop_unused else None, num_proc=num_proc,
         desc=f"process_dataset: q_latent_{scheduled_stage}"
    ).with_format('pt')


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    bot_id,
    latent_id,
    eot_id,
    shuffle=False,
    drop_unused=True,
    num_proc=default_num_proc,
    skip_bot_eot=True,
):
    """chain of thought latent dataset for training
    
    format: question, latent, reasoning, answer
    """
    no_bot_eot = scheduled_stage < 0 or skip_bot_eot
    # Number of wrapper tokens around latent block: [bot_id] + latents + [eot_id]
    # These need to be masked in labels (set to -100) along with question and latent tokens
    n_additional_tokens = 0 if no_bot_eot else 2

    def process_dataset(sample):
        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = max(0, scheduled_stage)

        # progressivly replace reasoning steps with latent tokens
        # n_skip_steps: number of reasoning steps to skip, replace with `c_thought` latent tokens
        # if scheduled_stage_to_train <= configs.max_latent_stage:
        #      n_skip_steps, n_latent_tokens = (
        #         scheduled_stage_to_train,
        #         scheduled_stage_to_train,
        #     )
        # else:
        n_skip_steps = 10000  # skip all verbal reasoning steps
        if configs.pad_latent_to_max:
            n_latent_tokens = configs.max_latent_stage
        else:
            n_latent_tokens = min(
                len(sample["steps_tokenized"]), configs.max_latent_stage
            )
        
        # if configs.no_cot:
        #     n_skip_steps = 100  # skip all step
        #     n_latent_tokens = 0

        # for each reasoning step we use X tokens
        n_latent_tokens *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_bot_eot else [bot_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_bot_eot else [eot_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )


        labels = [-100] * (
            len(tokens)
        )
        answer_labels = torch.tensor(sample['answer_mask']) * torch.tensor(sample["answer_tokenized"])
        labels[-len(answer_labels):] = answer_labels.tolist()

        return {
            "input_ids": tokens,
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],  
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            # "position_ids": list(range(len(tokens))),
        }

    processed_dataset = base_dataset.map(
        process_dataset, remove_columns=list(base_dataset.features) if drop_unused else None, num_proc=num_proc,
        desc=f"process_dataset: cot_latent_{scheduled_stage}"
    )
    if shuffle:
        processed_dataset = processed_dataset.shuffle()
    dataset = processed_dataset.with_format('pt')

    return dataset
