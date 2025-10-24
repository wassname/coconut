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

from coconut.trmlora.recursive_lora import TRMConfig, TRMLoraModel
from peft import PeftModel, get_peft_model

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
    conf.eos_token_id = tokenizer.eos_token_id

    # load base model
    model_config = AutoConfig.from_pretrained(
        conf.model_id,
        latent_token_id=latent_id,
        bot_id=bot_id,
        eot_id=eot_id,
        eos_token_id=tokenizer.eos_token_id,
        use_position_ids=conf.use_position_ids,
    )
    
    # Load with quantization if specified (for TRM or LoRA)
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
    elif getattr(conf, 'use_trm', False):
        logger.info("TRM mode: loading model with quantization")
        quantization_config = None  # Default no quant for TRM if not specified

    base_model = AutoModelForCausalLM.from_pretrained(
        conf.model_id, config=model_config, device_map=device, torch_dtype=dtype, quantization_config=quantization_config
    )

    base_model.resize_token_embeddings(len(tokenizer))

    logger.info("Loading TRM LoRA adapter")

    num_layers = base_model.config.num_hidden_layers
    # trm_hidden_size = base_model.config.hidden_size

    
    target_layers = [int(x/num_layers) for x in range(0, 100, 10)]
    target_modules = [k for k,v in base_model.named_modules() if (".mlp." in k) and isinstance(v, torch.nn.Linear) and any(f".{i}." in k for i in target_layers)]
    logger.info(f"Targeting {len(target_modules)} modules for TRM LoRA adapters: {target_modules}")
    peft_config = TRMConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=conf.lora_r,
        lora_alpha=conf.lora_alpha,
        lora_dropout=conf.lora_dropout,
        # target_modules="all-linear",  # Target all linear layers
        target_modules=target_modules,

        l_cycles=conf.trm_l_cycles,
        h_cycles=conf.trm_h_cycles,
        # hidden_size=trm_hidden_size,
        # llm_hidden_size=base_model.config.hidden_size,
        expansion=conf.trm_expansion,
        l_layers=conf.trm_l_layers,
        num_heads=conf.trm_num_heads,
        update_mode='lora',
        bias="none",
        modules_to_save=None,
    )
    peft_model = get_peft_model(base_model, peft_config)
    peft_model = peft_model.to(device)
    peft_model.enable_input_require_grads()
    # OR 
    # peft_model = TRMLoraModel(base_model, peft_config, "default")

    peft_model.print_trainable_parameters()

    model = Coconut(peft_model, conf)
    return model, tokenizer

def resume_model(conf: BaseConfig, device="auto", dtype=torch.bfloat16):

    model, tokenizer = load_new_model(conf, device, dtype)

    if getattr(conf, 'use_trm_lora', False):
        logger.info("Loading TRM LoRA adapter weights")
        model.model.load_adapter(conf.load_model_path)
        logger.info(f"Resumed TRM LoRA adapter from {conf.load_model_path}")
    else:
        state_dict = safetensors.torch.load_file(conf.load_model_path, device=device)
        model.load_state_dict(state_dict, strict=False)
        logger.warning(f"Resumed model from {conf.load_model_path}")

    # set the configuration
    return model, tokenizer


def save_model(model, tokenizer, configs, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "coconut_config.toml", "w") as f:
        toml.dump(configs, f)
    logger.info(f"saving model {save_dir}")

    # save state dict (only TRM adapter, not frozen base model)
    state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.model.')}
    save_folder = str(save_dir / "trmlora/")
    logger.error("FIXME save with custom peft type: Unknown PEFT type passed: TRMLORA")
    # model.model.save_pretrained(save_folder)
    logger.info(f"saving model {save_folder}")
