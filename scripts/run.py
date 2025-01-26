import argparse
import gc
import json
import os
import sys
from copy import copy
import pandas as pd
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
    get_question_only_latent_dataset,
)
from coconut.utils import Config, set_seed
from pathlib import Path
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
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

def save_model(model, f):
    states = model.state_dict()
    torch.save(states, f)
    logger.info(f"saving model. {f}")        

def main():
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

        logger.info(f"Config: {config_dict}")

    configs = Config(config_dict)

    if os.environ.get('DEBUG', False):
        configs.debug = True
        logger.warning("Debug mode is on")

    set_seed(configs.seed)
    save_dir = Path(configs.save_path) / configs.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # checkpoints = [f.name for f in save_dir.glob("checkpoint_*")]
    # checkpoints = sorted(checkpoints, key=lambda x: int(x.stem.split("_")[1]))

    # load base model
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, bot_id, eot_id]:
        # tie embeddings for special tokens
        target_embedding = embeddings.weight.data[token_id]
        embeddings.weight.data[token_id] = target_embedding.clone()
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id].clone()

    # set devices
    print_cuda_devices()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if configs.bf16 else torch.float32
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model = model.to(device)

    model = Coconut(model, latent_id, bot_id, eot_id, tokenizer.eos_token_id)

    # setup eval
    logger.info(model)
    max_size=32 if configs.debug else (configs.max_size or 100000000)
    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=max_size, drop_unused=False
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=max_size
        )

    # wandb
    if not configs.debug and not configs.only_eval:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb_run = None

    collator = CoconutCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # Set up training arguments
    training_args = TrainingArguments(
        run_name=configs.name,
        output_dir=save_dir,
        per_device_train_batch_size=configs['batch_size_training'],
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        learning_rate=configs['lr'],
        warmup_ratio=0.1,
        # max_steps=configs['samples_per_epoch']//configs['batch_size_training']*configs['num_epochs'],
        logging_steps=100, # TODO ideally we log to tensorboard every step, but to ui every 100 steps
        save_steps=10000,
        bf16=configs.bf16,
        bf16_full_eval=configs.bf16,
        optim="adamw_torch", # save memory: adamw_bnb_8bit
        num_train_epochs=1,#configs['epochs_per_stage'],
        torch_empty_cache_steps=100,
        save_safetensors=False,
        report_to="wandb" if wandb_run else None,
        lr_scheduler_type="cosine",# cosine cosine_with_restarts
        
        # save_strategy="no",
    )

    for phase in range(2):
        for epoch in range(configs.num_epochs if (phase>0) else 1):
            if phase==0:
                scheduled_stage = 0
                configs.c_thought=0
                configs.max_latent_stage = 0
                configs.cot = True
                configs.coconut = False
            else:
                scheduled_stage = epoch // configs['epochs_per_stage']
                configs.c_thought = 2
                configs.max_latent_stage = 3
                configs.cot = False
                configs.coconut = True

            no_bot_eot=(scheduled_stage == 0)


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
            # if epoch==0:
            #     r = evaluate(valid_gen_dataloader, model, tokenizer, base_dataset_valid, max_new_tokens=max_new_tokens, name=f"eval_{phase}_{epoch}_start")
            #     if wandb_run:
            #         wandb_run.log(r)

            logger.info(f"Training stage {scheduled_stage}, phase {phase}")

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

                trainer = Trainer(
                    model=model if scheduled_stage > 0 else model.base_causallm,
                    args=training_args,
                    train_dataset=dataset_train,
                    eval_dataset=dataset_loss_val,
                    data_collator=collator,
                )
                # TODO we don't need to shuffle train as it's done during load
                clear_memory()
                trainer.train()


            clear_memory()
            r = evaluate(valid_gen_dataloader, model, tokenizer, base_dataset_valid, max_new_tokens=max_new_tokens, name=f"eval_{phase}_{epoch}")
            clear_memory()
            if wandb_run:
                wandb_run.log(r)

            save_model(model, save_dir / f"checkpoint_{phase}_{epoch}.pt")
        


def evaluate(dataloader, model, tokenizer, ds, max_new_tokens=64, device='cuda', name=""):


    # get original answer
    question_val = ds["question"]
    answers_val = [
        d.replace(",", "").strip() for d in ds["answer"]
    ]
    cot_val = ["\n".join(d) for d in ds["steps"]]

    # val generation accuracy
    total_length = len(dataloader)

    pbar = tqdm(
        colour="green", desc="Test Accuracy", total=total_length, dynamic_ncols=True
    )
    logger.info(f"Starting evaluation {name}")
    cor, cor_cot, total = 0, 0, 0
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(dataloader):
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

            # TODO use amp?

            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

            def indent(s):
                return s.replace("\n", "\n\t")

            llm_text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            llm_answer_output = llm_text_output.split("#")[-1].replace(",", "").strip()
            llm_cot_output = (
                ("\n".join(llm_text_output.split("\n")[1:])).split("#")[0].strip()
            )

            if idx < 5:
                correct = '✅' if llm_answer_output==answer else '❌'
                logger.info(
                    f"""Question {test_idx}: Answer = '{answer}' CoT = '{indent(answer_cot)}'
Extracted llm Output: '{llm_answer_output}' (=? {answer}) {correct}.
Full llm output: '{indent(tokenizer.decode(outputs[0]))}'. 
""")
                

            cor += llm_answer_output == answer
            cor_cot += llm_cot_output == answer_cot

            pbar.update(1)
            pbar.set_description(f"Test accuracy: {round(float(cor / total), 2)}")

        pbar.close()
        logger.info(f"Cor={cor}, CoT={cor_cot}, Total={total}")
        logger.info(f"Accuracy on validation set:  {cor} / {total} = {cor / total}")
        logger.info(
            f"CoT match on validation set: {cor_cot} / {total} = {cor_cot / total}"
        )

    return {"eval/acc": cor / total, "eval/cot_em": cor_cot / total}


if __name__ == "__main__":
    main()
