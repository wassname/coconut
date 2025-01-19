import argparse
import gc
import json
import os
import sys
from copy import copy

from loguru import logger
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from coconut.coconut import Coconut
from coconut.dataset import (
    CoconutCollator,
    get_cot_latent_dataset,
    get_dataset,
    get_question_latent_dataset,
)
from coconut.utils import Config, set_seed
from pathlib import Path

def print_cuda_devices():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()): 
            torch.cuda.get_device_name(i)
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(torch.cuda.get_device_capability(i))
            print(torch.cuda.get_device_properties(i))
    else:
        logger.warning("CUDA is not available")


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def save_model(model, f):
    states = model.state_dict()
    torch.save(states, f)
    print(f"saving model. {f}")

def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

        logger.info(f"Config: {config_dict}")

    configs = Config(config_dict)

    if os.environ['DEBUG']:
        configs.debug = True
        logger.warning("Debug mode is on")

    set_seed(configs.seed)
    save_dir = Path(configs.save_path) / configs.name
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [f.name for f in save_dir.glob("checkpoint_*")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.stem.split("_")[1]))


    # check if the job is preempted and resumed.
    if len(checkpoints) > 0 and not configs.only_eval:

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = save_dir / latest_checkpoint

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        if configs.load_model_path == "None":
            logger.warning(
                f"You want to skip the first {configs.resume} epochs but you are not loading existing checkpoint!"
            )
        logger.info(f"Loading from {configs.load_model_path}. Skipping the first {configs.resume} epochs")

    # load base model
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(configs.load_model_path)
        if configs.coconut and not any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and any(
            [k.startswith("base_causallm") for k in saved_weights.keys()]
        ):
            pass

        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        target_id = tokenizer.convert_tokens_to_ids("<<")
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[token_id]
            embeddings.weight.data[token_id] = target_embedding
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    print_cuda_devices()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if configs.bf16 else torch.float32
    logger.info(f"Using device: {device}, dtype: {dtype}")
    use_amp = configs.bf16

    # torch.set_default_device(device)
    model = model.to(device)

    # if configs.bf16:
    #     model.to(torch.bfloat16)

    print(model)
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    if not configs.debug and not configs.only_eval:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])
    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None
    else:
        # adam = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # add bnb optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_acc = 0

    collator = CoconutCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):
        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
        )

        if not configs.only_eval:
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=True,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
            )

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
            )

            if configs.reset_optimizer:
                del optimizer
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            model.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch + 1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):
                if step == 0 and wandb_run:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(device) for key in batch.keys() if key != "idx"
                }

                with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
                    outputs = model(**batch)

                    loss = outputs.loss / configs.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                # loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch + 1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                f = os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                save_model(model, f)

                clear_memory()

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

                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                if wandb_run:
                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        # val generation accuracy
        total_length = len(valid_gen_dataloader)

        pbar = tqdm(
            colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
        )
        cor, cor_cot, total = 0, 0, 0
        with torch.no_grad():
            model.eval()
            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if v != None and k not in ["idx", "position_ids"]
                }
                assert len(batch["input_ids"]) == 1
                answer = answers_val[test_idx.item()]
                answer_cot = cot_val[test_idx.item()]
                question = question_val[test_idx.item()]

                total += 1

                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = (
                    ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
                )

                if idx < 5:
                    print(
                        f"Question {test_idx}: Answer = '{answer}' CoT = '{answer_cot}'"
                    )
                    print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted Output: '{answer_output}'")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(f"Test accuracy: {round(float(cor / total), 2)}")

            pbar.close()
            print(f"Cor={cor}, CoT={cor_cot}, Total={total}")

            print(f"Accuracy on validation set: {cor} / {total} = {cor / total}")
            print(
                f"CoT match on validation set: {cor_cot} / {total} = {cor_cot / total}"
            )
        sys.stdout.flush()

        if wandb_run:
            wandb_run.log({"eval/acc": cor / total, "eval/cot_em": cor_cot / total})

        if configs.only_eval:
            break

        if (
            cor / total > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            f = os.path.join(save_dir, f"checkpoint_{epoch + 1}")
            save_model(model, f)

            best_acc = cor / total
            clear_memory()


if __name__ == "__main__":
    main()
