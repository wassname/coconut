import argparse
import gc
import json
import os
import time
from copy import copy
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
import yaml
from loguru import logger
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)

import wandb
from coconut.coconut import (
    CoconutConfig,
    CoconutQwen3ForCausalLM,
)
from coconut.dataset import (
    CoconutCollator,
    get_cot_latent_dataset,
    get_dataset,
    get_question_only_latent_dataset,
)
from coconut.eval import evaluate
from coconut.utils import Config, convert_to_bfloat16, set_seed

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


def create_optimizer(model, configs, warmup_steps=False):
    if configs.bf16_weight:
        import optimi

        optimizer = optimi.AdamW(
            model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif configs.opt_8b:
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam8bit(
            model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    if warmup_steps is not None:
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )
    return optimizer, scheduler


def main():
    from coconut import configs # this will be my dataclass files
    experiments = configs.__dict__.keys()
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("experiment", type=str, help=f"experiment names: [{experiments}]")
    args = parser.parse_args()

    conf = getattr(configs, args.experiment)

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
    device = "cuda"  # if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if conf.bf16 else torch.float32
    logger.info(f"Using device: {device}, dtype: {dtype}")

    if conf.resume_epochs:
        f = Path("./" + conf.load_model_path)
        assert f.exists(), f"Model path {f} does not exist"
        model = CoconutQwen3ForCausalLM.from_pretrained(
            conf.load_model_path, device_map="auto", torch_dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(conf.load_model_path)
        logger.warning(f"Loaded model from {conf.load_model_path}")
    else:
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            conf.model_id,
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
        model_config = CoconutConfig.from_pretrained(
            conf.model_id,
            latent_token_id=latent_id,
            bot_id=bot_id,
            eot_id=eot_id,
            eos_token_id=tokenizer.eos_token_id,
            replacement_method=conf.replacement_method,
        )
        model = CoconutQwen3ForCausalLM.from_pretrained(
            conf.model_id, config=model_config, device_map=device, torch_dtype=dtype
        )

        model.resize_token_embeddings(len(tokenizer))

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    # TODO check this is in vocab
    for token_id in [latent_id, bot_id, eot_id]:
        # tie embeddings for special tokens
        target_embedding = embeddings.weight.data[target_id]
        embeddings.weight.data[token_id] = target_embedding.clone()

        # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id].clone()

    model = model.to(device)

    if conf.bf16_weight:
        convert_to_bfloat16(model)

    # setup eval
    logger.info(model)
    max_size = 32 if conf.debug else (conf.max_size or 100000000)
    base_dataset_valid = get_dataset(
        conf.val_path,
        tokenizer,
        max_size=max_size // 30 + 3,
        drop_unused=False,
        system_prompt=conf.system_prompt,
    )
    # print(configs.get("system_prompt", ""), "system_prompt")
    # logger

    if not conf.only_eval:
        base_dataset_train = get_dataset(
            conf.train_path, tokenizer, max_size=max_size
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

    logger.info(
        f"loading with optimisations: 8 bit gradients: {conf.opt_8b}, brainfloat 16 inputs: {conf.bf16_weight}, bf16 model weights with  Kahan Summation optimiser (experimental): {conf.bf16}"
    )

    optimiser = None
    collator = CoconutCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    """
    The stages
    - phase 0: epoch 0: normal CoT training, with bot and eot tokens, to get it used to the structure with no recusion yet
    - phase 1+: epoch N: CoT training, with bot and eot tokens, add X more <latent> tokens each stage, until you have X times more than steps in the original dataset
    """

    res = []

    if conf.resume_epochs:
        logger.warning(f"Resuming from epoch {conf.resume_epochs}")

    for epoch in tqdm(range(conf.resume_epochs, conf.num_epochs), unit="epoch"):
        start_time = time.time()

        max_latent_epoch = conf.max_latent_stage * conf.epochs_per_stage

        if epoch <= conf.cot_epochs:
            scheduled_stage = 0
            no_bot_eot = True
        elif epoch < max_latent_epoch:
            scheduled_stage = (epoch - conf.cot_epochs) // conf.epochs_per_stage
            no_bot_eot = False
        else:
            scheduled_stage = conf.max_latent_stage

        logger.info(
            f"scheduled_stage={scheduled_stage}, no_bot_eot={no_bot_eot}, c_thought={conf.c_thought}, max_latent_stage={conf.max_latent_stage}"
        )

        # initial eval
        dataset_gen_val = get_question_only_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            conf,
            bot_id,
            latent_id,
            eot_id,
            no_bot_eot=no_bot_eot,
            # drop_unused=False,
        )
        if "gsm" in conf.val_path:
            max_new_tokens = 64
        else:
            max_new_tokens = 128
        if conf.debug:
            max_new_tokens = 8
            print("DEBUG MODE: max_new_tokens set to 8")
        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=6,
            pin_memory=True,
            batch_size=conf.batch_size_training,
            collate_fn=collator,
        )
        if epoch == 0:
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
                quick=True,
            )
            if wandb_run:
                wandb_run.log(r)

        logger.info(f"Prep data for epoch={epoch} stage={scheduled_stage}")

        dataset_loss_val = get_cot_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            conf,
            bot_id,
            latent_id,
            eot_id,
            no_bot_eot=no_bot_eot,
        )
        valid_loss_dataloader = torch.utils.data.DataLoader(
            dataset_loss_val,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=conf.batch_size_training,
            collate_fn=collator,
        )

        if not conf.only_eval:
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                conf,
                bot_id,
                latent_id,
                eot_id,
                no_bot_eot=no_bot_eot,
                shuffle=True,
            )
            if conf.reset_optimizer or optimiser is None:
                warmup_steps = None
                if epoch == 0 or epoch == (conf.cot_epochs + 1):
                    warmup_steps = (
                        len(dataset_train) // conf.gradient_accumulation_steps * 0.1
                    )

                optimizer, scheduler = create_optimizer(
                    model, conf, warmup_steps=warmup_steps
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

                    for k in outputs.log:
                        if wandb_run:
                            wandb_run.log({f"train/{k}": outputs.log[k]})

                    loss = outputs.loss / conf.gradient_accumulation_steps

                loss.backward()

                norm = None
                # # every N steps (or last batch) do optimizer step
                is_last_step = step == len(train_dataloader) - 1
                if (
                    step + 1
                ) % conf.gradient_accumulation_steps == 0 or is_last_step:
                    if conf.grad_clip is not None:
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
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"T Epoch: {epoch}/{conf.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"(loss: {round(float(loss.detach().float() * conf.gradient_accumulation_steps), 4):2.2f}"
                )
                if step % 100 == 0:
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
                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

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
        r["phase"] = epoch
        r["minutes"] = (time.time() - start_time) / 60
        clear_memory()
        if wandb_run:
            wandb_run.log(r)
        res.append(r)

        save_model(model, tokenizer, config_dict, save_dir / f"checkpoint_{epoch}")

    print(f"\n# Results: {run_name}")
    print(config_dict)
    df_res = pd.DataFrame(res)
    df_res.to_csv(save_dir / "results.csv")
    print(df_res.to_markdown())


if __name__ == "__main__":
    main()
