import argparse
import gc
import json
import os
import sys
from copy import copy
import pandas as pd
import time
import torch
from torch import nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from coconut.dataset import (
    CoconutCollator,
    get_cot_latent_dataset,
    get_dataset,
    get_question_only_latent_dataset,
)
from coconut.coconut import (
    CoconutConfig,
    CoconutQwen2ForCausalLM,
)
from coconut.eval import evaluate
from pathlib import Path

from coconut.trainer_optimi import TrainerOptimi, convert_to_bfloat16


import transformers
from coconut.utils import Config, set_seed, ProgressCallbackNoPrint, rm_old_prog_cb
transformers.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNoPrint # monkey patch the default progress callback
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    TrainerCallback,
)

from loguru import logger
logger.remove()
def sink(msg):
    return tqdm.write(msg, end="")
logger.add(sink, colorize=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

def save_model(model, tokenizer, configs, save_dir: Path):
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(configs, f)
    logger.info(f"saving model {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

        logger.info(f"Config: {config_dict}")

    configs = Config(config_dict)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{configs.name}_{timestamp}"

    if os.environ.get('DEBUG', False):
        configs.debug = True
        logger.warning("Debug mode is on")

    set_seed(configs.seed)
    save_dir = Path(configs.save_path) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # set devices
    print_cuda_devices()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if configs.bf16 else torch.float32
    logger.info(f"Using device: {device}, dtype: {dtype}")

    if configs.load_model_path:
        f = Path('./' + configs.load_model_path)
        assert f.exists(), f"Model path {f} does not exist"
        model = CoconutQwen2ForCausalLM.from_pretrained(configs.load_model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(configs.load_model_path)
        logger.warning(f"Loaded model from {configs.load_model_path}")
    else:
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(configs.model_id,
                                                padding_side="right",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if "<|latent|>" not in tokenizer.additional_special_tokens:
            # model.generation_config.pad_token_id = tokenizer.pad_token_id
            tokenizer.add_tokens("<|start-latent|>")
            tokenizer.add_tokens("<|end-latent|>")
            tokenizer.add_tokens("<|latent|>")

        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

        # load base model
        model_config = CoconutConfig.from_pretrained(configs.model_id,    latent_token_id=latent_id, 
            bot_id=bot_id, eot_id=eot_id, eos_token_id=tokenizer.eos_token_id, replacement_method=configs.replacement_method)
        model = CoconutQwen2ForCausalLM.from_pretrained(configs.model_id, config=model_config, device_map=device)
        
        model.resize_token_embeddings(len(tokenizer))


    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, bot_id, eot_id]:
        # tie embeddings for special tokens
        target_embedding = embeddings.weight.data[token_id]
        embeddings.weight.data[token_id] = target_embedding.clone()
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id].clone()

    model = model.to(device)
    
    if configs.bf16_weight:
        convert_to_bfloat16(model)

    # setup eval
    logger.info(model)
    max_size=32 if configs.debug else (configs.max_size or 100000000)
    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=max_size//30+3, drop_unused=False
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=max_size
        )

    # wandb
    if not configs.debug and not configs.only_eval:
        wandb_run = wandb.init(project=configs.project, group=configs.name, 
                               name=run_name,
                            #    resume="allow"
                               )
        wandb_run.config.update(configs, allow_val_change=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb_run = None

    collator = CoconutCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # Set up training arguments
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=save_dir,
        per_device_train_batch_size=configs['batch_size_training'],
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        learning_rate=configs['lr'],
        warmup_ratio=0.2,
        # max_steps=configs['samples_per_epoch']//configs['batch_size_training']*configs['num_epochs'],
        logging_steps=1, # TODO ideally we log to tensorboard every step, but to ui every 100 steps
        save_steps=10000,
        bf16=configs.bf16,
        bf16_full_eval=configs.bf16,
        optim="adamw_bnb_8bit", # save memory:adamw_torch  adamw_bnb_8bit or paged_adamw_32bit
        num_train_epochs=1,#configs['epochs_per_stage'],
        torch_empty_cache_steps=100,
        save_safetensors=False,
        save_only_model=True,
        report_to="wandb" if wandb_run else None,
        # lr_scheduler_type="cosine",# cosine cosine_with_restarts
        
        # save_strategy="no",
    )

    """
    The stages
    - phase 0: epoch 0: normal CoT training, with bot and eot tokens, to get it used to the structure with no recusion yet
    - phase 1+: epoch N: CoT training, with bot and eot tokens, add X more <latent> tokens each stage, until you have X times more than steps in the original dataset
    """
    

    res = []

    if configs.resume:
        logger.warning(f"Resuming from epoch {configs.resume}")
    for phase in range(configs.resume, configs.num_epochs):
        start_time = time.time()
        if phase == configs.max_latent_stage:
            training_args.num_train_epochs = configs.num_epochs - configs.max_latent_stage
            print("max_latent_stage reached, training in one large run for", training_args.num_train_epochs)
        elif phase > configs.max_latent_stage:
            break

        scheduled_stage = phase // configs['epochs_per_stage']
        no_bot_eot=configs.cot or configs.no_cot or configs.no_thoughts
        logger.info(f"scheduled_stage={scheduled_stage}, no_bot_eot={no_bot_eot}, c_thought={configs.c_thought}, max_latent_stage={configs.max_latent_stage}, cot={configs.cot}, coconut={configs.coconut}")

        # initial eval
        dataset_gen_val = get_question_only_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            bot_id,
            latent_id,
            eot_id,
            no_bot_eot=no_bot_eot,
            # drop_unused=False,
        )
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
        )
        if "gsm" in configs.val_path:
            max_new_tokens = 64
        else:
            max_new_tokens = 128
        if phase==0:
            # quick QC to see how well untouched model does at the task
            r = evaluate(valid_gen_dataloader, model, tokenizer, base_dataset_valid, max_new_tokens=max_new_tokens, name=f"eval_{phase}_start", dtype=dtype, device=device, quick=True)
            if wandb_run:
                wandb_run.log(r)

        logger.info(f"Training stage epoch={phase} stage={scheduled_stage}")

        dataset_loss_val = get_cot_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            bot_id,
            latent_id,
            eot_id,
            no_bot_eot=no_bot_eot,
        )

        if not configs.only_eval:
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                bot_id,
                latent_id,
                eot_id,
                no_bot_eot=no_bot_eot,
                shuffle=True,
            )

            if configs.bf16_weight:
                TrainerCls = TrainerOptimi
            else:
                TrainerCls = Trainer

            class CoconutEvalCallback(TrainerCallback):               
                
                def on_epoch_end(self, *args, **kwargs):
                    r = evaluate(valid_gen_dataloader, model, tokenizer, base_dataset_valid, max_new_tokens=max_new_tokens, name=f"eval_{phase}", dtype=dtype, device=device)
                    if wandb_run:
                        wandb_run.log(r)
                    res.append(r)
                    return 
            trainer = TrainerCls(
                model=model,# if scheduled_stage > 0 else model.base_causallm,
                args=training_args,
                train_dataset=dataset_train,
                eval_dataset=dataset_loss_val,
                data_collator=collator,
                callbacks=[ProgressCallbackNoPrint(), CoconutEvalCallback()]
                # TODO pass in (opt, scheduler) as a callback
            )
            # TODO we don't need to shuffle train as it's done during load
            clear_memory()
            rm_old_prog_cb(trainer)
            try:
                trainer.train()
            except KeyboardInterrupt:
                logger.info("Interrupted")
                pass

        clear_memory()
        r = evaluate(valid_gen_dataloader, model, tokenizer, base_dataset_valid, max_new_tokens=max_new_tokens, name=f"eval_{phase}", dtype=dtype, device=device)
        r['epoch'] = phase
        r['minutes'] = (time.time() - start_time) / 60
        clear_memory()
        if wandb_run:
            wandb_run.log(r)
        res.append(r)

        save_model(model, tokenizer, config_dict, save_dir / f"checkpoint_{phase}")

    print(f'\n# Results: {run_name}')
    print(config_dict)
    df_res = pd.DataFrame(res)
    df_res.to_csv(save_dir / "results.csv")
    print(df_res.to_markdown())
        


if __name__ == "__main__":
    main()
