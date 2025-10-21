from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, AutoConfig,
    get_constant_schedule_with_warmup,
)
from coconut.coconut import Coconut
from coconut.configs import BaseConfig
from loguru import logger
from pathlib import Path
import torch
import safetensors.torch
import toml
from transformers import BitsAndBytesConfig

def load_new_model(conf: BaseConfig, device, dtype):
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

    conf.latent_token_id = latent_id
    conf.bot_token_id = bot_id
    conf.eot_token_id = eot_id

    # load base model
    model_config = AutoConfig.from_pretrained(
        conf.model_id,
        latent_token_id=latent_id,
        bot_id=bot_id,
        eot_id=eot_id,
        eos_token_id=tokenizer.eos_token_id,
        use_position_ids=conf.use_position_ids,
    )
    
    # TRM mode: load with quantization
    if getattr(conf, 'use_trm', False):
        logger.info("TRM mode: loading model with quantization")
        
        quantization_config = None
        if getattr(conf, 'load_in_4bit', False):
            logger.info("Loading in 4bit")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif getattr(conf, 'load_in_8bit', False):
            logger.info("Loading in 8bit")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        conf.model_id, config=model_config, device_map=device, torch_dtype=dtype, quantization_config=quantization_config
    )
    
    # apply_config(model, tokenizer, conf)

    base_model.resize_token_embeddings(len(tokenizer))

    model = Coconut(base_model, conf)
    return model, tokenizer

def resume_model(conf: BaseConfig, device="auto", dtype=torch.bfloat16):

    model, tokenizer = load_new_model(conf, device, dtype)

    state_dict = safetensors.torch.load_file(conf.load_model_path, device=device)
    model.load_state_dict(state_dict, strict=False)
    logger.warning(f"Resumed model from {conf.load_model_path}")

    # set the configuration
    return model, tokenizer

def tie_embeddings(base, tokenizer):
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    # tie the embeddings for the special tokens
    embeddings = base.model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    # TODO check this is in vocab
    for token_id in [latent_id, bot_id, eot_id]:
        # tie embeddings for special tokens
        target_embedding = embeddings.weight.data[target_id]
        embeddings.weight.data[token_id] = target_embedding.clone()

        # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
        lm_head = base.model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id].clone()
    return base

def save_model(model, tokenizer, configs, save_dir: Path):
    # tokenizer.save_pretrained(save_dir)
    # model.model.save_pretrained(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "coconut_config.toml", "w") as f:
        toml.dump(configs, f)
    logger.info(f"saving model {save_dir}")

    # save state dict (only TRM adapter, not frozen base model)
    state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.model.')}
    safetensors.torch.save_file(state_dict, str(save_dir / "pytorch_model.safetensors"))
    logger.info(f"saving model {save_dir / 'pytorch_model.safetensors'}")
