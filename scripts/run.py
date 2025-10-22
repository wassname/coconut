import argparse
# Early warning filter: suppress noisy Pydantic UnsupportedFieldAttributeWarning
# which can be emitted when generating schemas for dataclasses used by tyro.
import warnings
from torchinfo import summary
try:
    from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:
    # Fallback: filter by message substrings if pydantic internals path changes
    warnings.filterwarnings(
        "ignore",
        message="The 'repr' attribute with value False was provided to the `Field()` function",
    )
    warnings.filterwarnings(
        "ignore",
        message="The 'frozen' attribute with value True was provided to the `Field()` function",
    )

import gc
import json
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
    CoconutCollator,
    get_cot_latent_dataset,
    get_dataset,
    get_question_only_latent_dataset,
)
from coconut.eval import evaluate, get_answer_perplexity, get_answer_preference
from coconut.utils import Config, convert_to_bfloat16, set_seed, clear_memory, print_cuda_devices
from coconut.load_model import (
    load_new_model,
    resume_model,
    tie_embeddings,
    save_model,
)

logger.remove()


def sink(msg):
    return tqdm.write(msg, end="")


logger.add(sink, colorize=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_ratio_eval(
    model,
    tokenizer,
    base_dataset_valid,
    conf,
    stage,
    device="cuda",
    dtype=torch.bfloat16,
):
    """helper function as we run this multiple times and it needs a diff val ds/dl."""
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    collator = CoconutCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    dataset_gen_val2 = get_cot_latent_dataset(
        stage,
        base_dataset_valid,
        conf,
        bot_id,
        latent_id,
        eot_id,
        # no_bot_eot=no_bot_eot,
        # drop_unused=False,
    )
    valid_gen_dataloader2 = torch.utils.data.DataLoader(
        dataset_gen_val2,
        num_workers=6,
        pin_memory=True,
        batch_size=conf.batch_size_training,
        collate_fn=collator,
    )
    r2 = get_answer_preference(
        model,
        tokenizer,
        valid_gen_dataloader2,
        device=device,
        dtype=dtype,
    )
    return r2



def create_optimizer(model, configs, warmup_fraction=0.1, opt_steps=None, cycles=1):
    warmup_steps = opt_steps * warmup_fraction
    scheduler = None
    # if configs.bf16_weight:
    #     import optimi

    #     optimizer = optimi.AdamW(
    #         model.parameters(),
    #         lr=configs.lr,
    #         weight_decay=configs.weight_decay if configs.weight_decay else None,
    #         kahan_sum=True,
    #     )
    # elif configs.opt_8b:
    #     import bitsandbytes as bnb

    #     optimizer = bnb.optim.Adam8bit(
    #         model.parameters(),
    #         lr=configs.lr,
    #         weight_decay=configs.weight_decay,
    #     )
    # else:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay,
    )
    if warmup_steps is not None:
        if configs.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=opt_steps,
            )
        elif configs.scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps
            )
        elif configs.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_cycles=cycles/2.,
                num_training_steps=opt_steps,
            )
    return optimizer, scheduler




def main():
    import tyro
    from coconut import configs # this will be my dataclass files
    experiments = configs.__dict__.keys()
    print(f"Available experiments: {experiments}")
    # parser = argparse.ArgumentParser(description="coconut")
    # parser.add_argument("experiment", type=str, help=f"experiment names: [{experiments}]")
    # args = parser.parse_args()

    print(os.sys.argv)
    ConfigCls = getattr(configs, os.sys.argv[1])
    conf = tyro.cli(ConfigCls, args=os.sys.argv[2:])
    config_dict = asdict(conf)
    logger.info(f"Config: {config_dict}")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{conf.name}_{timestamp}"

    if os.environ.get("DEBUG", False):
        conf.debug = True
        logger.warning("Debug mode is on")

    set_seed(conf.seed)
    save_dir = Path(conf.save_path) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # set devices
    print_cuda_devices()
    device = "cuda:0"  # if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (conf.bf16 is True) else torch.float32
    logger.info(f"Using device: {device}, dtype: {dtype}")

    if conf.resume_epochs>0 and conf.load_model_path:    
        logger.warning(f"Resuming from epoch {conf.resume_epochs}")    
        model, tokenizer = resume_model(conf, device, dtype)
    else:
        model, tokenizer = load_new_model(conf, device, dtype)
        tie_embeddings(model, tokenizer)
    model = model.to(device)

    # if conf.bf16_weight is True:
    #     convert_to_bfloat16(model)

    # setup eval
    logger.debug(model)
    summary(model, input_size=(4, 12), dtypes=[torch.long], depth=4)


    max_size = 32 if conf.debug is True else (conf.max_size or 100000000)
    base_dataset_valid = get_dataset(
        conf.val_path,
        tokenizer,
        max_size=max_size // 30 + 3,
        drop_unused=False,
        system_prompt=conf.system_prompt,
    )
    logger.info("System prompt: \n" + conf.system_prompt)
    # logger

    if not conf.only_eval:
        base_dataset_train = get_dataset(
            conf.train_path, tokenizer, max_size=max_size,
            system_prompt=conf.system_prompt, verbose=True
        )

    # wandb
    if not conf.debug and not conf.only_eval:
        wandb_run = wandb.init(
            project=conf.project,
            group=conf.name,
            name=run_name,
            #    resume="allow"
        )
        wandb_run.config.update(conf, allow_val_change=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb_run = None

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    optimiser = None
    collator = CoconutCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    """
    The stages
    - phase 0: epoch 0: normal CoT training, with bot and eot tokens, to get it used to the structure with no recusion yet
    - phase 1+: epoch N: CoT training, with bot and eot tokens, add X more <latent> tokens each stage, until you have X times more than steps in the original dataset
    """

    res = []

    if conf.resume_epochs>0:
        logger.warning(f"Resuming from epoch {conf.resume_epochs}")

    for epoch in tqdm(range(conf.resume_epochs, conf.num_epochs), unit="epoch"):
        start_time = time.time()

        max_latent_epoch = conf.cot_epochs + conf.max_latent_stage * conf.epochs_per_stage

        if epoch <= conf.cot_epochs:
            stage = -1
        elif epoch < max_latent_epoch:
            stage = (epoch - conf.cot_epochs) // conf.epochs_per_stage
        else:
            stage = conf.max_latent_stage

        logger.info(
            f"scheduled_stage={stage}, c_thought={conf.c_thought}, max_latent_stage={conf.max_latent_stage}"
        )

        # initial eval

        dataset_gen_val = get_question_only_latent_dataset(
            stage,
            base_dataset_valid,
            conf,
            bot_id,
            latent_id,
            eot_id,
            # drop_unused=False,
        )
        if "gsm" in conf.val_path:
            max_new_tokens = 64
        else:
            max_new_tokens = 128
        if conf.debug:
            max_new_tokens = 16
            print("DEBUG MODE: max_new_tokens set to 8")
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=6,
            pin_memory=True,
            batch_size=conf.batch_size_training,
            collate_fn=collator,
        )
        if epoch == 0 or (epoch==conf.resume_epochs) and conf.eval_first_epoch:
            # quick QC to see how well untouched model does at the task
            r = evaluate(
                valid_gen_dataloader,
                model,
                tokenizer,
                base_dataset_valid,
                max_new_tokens=max_new_tokens,
                name=f"eval_{epoch}_start",
                dtype=dtype,
                device=device,
                # quick=True,
            )
            # r = {f"eval/quick_{k}": v for k, v in r.items()}
            r2 = run_ratio_eval(
                model,
                tokenizer,
                base_dataset_valid,
                conf,
                stage,
                device=device,
                dtype=dtype,
            )
            if wandb_run:
                wandb_run.log(r)
                wandb_run.log(r2)
            
            r["epoch"] = -1
            r["stage"] = stage
            r['eval/ratios'] = r2['eval/ratios']
            res.append(r)

        logger.info(f"Prep data for epoch={epoch} stage={stage}")

        dataset_loss_val = get_cot_latent_dataset(
            stage,
            base_dataset_valid,
            conf,
            bot_id,
            latent_id,
            eot_id,
        )
        valid_loss_dataloader = torch.utils.data.DataLoader(
            dataset_loss_val,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=conf.batch_size_training,
            collate_fn=collator,
        )

        log_dict = None
        eval_log_dict = None

        if not conf.only_eval:
            dataset_train = get_cot_latent_dataset(
                stage,
                base_dataset_train,
                conf,
                bot_id,
                latent_id,
                eot_id,
                shuffle=True,
            )
            if (conf.reset_optimizer is True) or (optimiser is None):
                opt_steps=len(dataset_train) // conf.gradient_accumulation_steps
                if not conf.reset_optimizer:
                    opt_steps *= conf.num_epochs
                epochs=1 if conf.reset_optimizer else conf.num_epochs
                optimizer, scheduler = create_optimizer(
                    model, conf, warmup_fraction=0.1, opt_steps=opt_steps,
                    cycles=epochs
                )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=True,
                pin_memory=True,
                batch_size=conf.batch_size_training,
                collate_fn=collator,
                # sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            optimizer.zero_grad()
            model.train()
            total_length = len(train_dataloader) // conf.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch}",
                total=total_length,
                dynamic_ncols=True,
            )
            total_train_steps = 0

            for step, batch in enumerate(train_dataloader):
                total_train_steps += 1
                batch = {
                    key: batch[key].to(device) for key in batch.keys() if key != "idx"
                }

                with torch.autocast(device_type=device, dtype=dtype):
                    outputs = model(**batch)

                    loss = outputs.loss / conf.gradient_accumulation_steps

                loss.backward()

                norm = None
                # # every N steps (or last batch) do optimizer step
                is_last_step = step == len(train_dataloader) - 1
                if (
                    step + 1
                ) % conf.gradient_accumulation_steps == 0 or is_last_step:
                    if (conf.grad_clip is not None) and (conf.grad_clip > 0):
                        norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), conf.grad_clip
                        )

                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

                    pbar.update(1)

                if wandb_run:
                    lr = torch.tensor(
                        [group["lr"] for group in optimizer.param_groups]
                    ).mean()
                    log_dict = {
                        "train/epoch": epoch,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * conf.gradient_accumulation_steps,
                        "train/lr": lr,
                        "train/grad_norm": norm,
                        **{f'train/{k}': outputs.log[k] for k in outputs.log},
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"T Epoch: {epoch}/{conf.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"(loss: {round(float(loss.detach().float() * conf.gradient_accumulation_steps), 4):2.2f}"
                )
                if step % 5 == 0:
                    clear_memory()
            pbar.close()

            # val loss
            total_loss = 0
            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(valid_loss_dataloader):
                    batch = {
                        key: batch[key].to(device)
                        for key in batch.keys()
                        if key != "idx"
                    }

                    with torch.autocast(device_type=device, dtype=dtype):
                        outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()

                if wandb_run:
                    eval_loss = total_loss / len(valid_loss_dataloader)
                    eval_perplexity = torch.exp(torch.tensor(eval_loss)).item()  # Absolute perplexity
                    eval_log_dict = {
                        "eval/loss": eval_loss,
                        "eval/perplexity": eval_perplexity,  # Track absolute confidence
                        **{f'eval/{k}': outputs.log[k] for k in outputs.log},
                    }
                    wandb_run.log(eval_log_dict)
                    
                    print("eval loss", eval_loss)
                    print("eval perplexity", eval_perplexity)
                    
                    # What to look for:
                    # - Perplexity decreasing: Model gaining confidence on predictions
                    # - Perplexity increasing while loss decreases: Potential overconfidence/miscalibration
                    # - Stable low perplexity with improving ratios: Good sign of latent reasoning

            clear_memory()

        clear_memory()
        r = evaluate(
            valid_gen_dataloader,
            model,
            tokenizer,
            base_dataset_valid,
            max_new_tokens=max_new_tokens,
            name=f"eval_{epoch}",
            dtype=dtype,
            device=device,
        )


        r2 = run_ratio_eval(
            model,
            tokenizer,
            base_dataset_valid,
            conf,
            stage,
            device=device,
            dtype=dtype,
        )
        # r3 = get_answer_perplexity(
        #     model,
        #     tokenizer,
        #     valid_gen_dataloader,
        #     dtype=dtype,
        #     device=device,
        # )
        # r['eval/ppx'] = r3['eval/ppx']
        r['eval/ratios'] = r2['eval/ratios']
        r["epoch"] = epoch
        r["stage"] = stage
        r["train/minutes"] = (time.time() - start_time) / 60
        clear_memory()
        if wandb_run:
            wandb_run.log(r)

        if log_dict is not None:
            r["train/loss"] = log_dict.get("train/loss", None)
        if eval_log_dict is not None:
            r['eval/loss'] = eval_log_dict.get("eval/loss", None)
        res.append(r)

        save_model(model, tokenizer, config_dict, save_dir / f"checkpoint_{epoch}")

    print(f"\n# Results: {run_name}")
    print(config_dict)
    df_res = pd.DataFrame(res)
    df_res.to_csv(save_dir / "results.csv")
    print(df_res.round(4).to_markdown())


if __name__ == "__main__":
    main()
