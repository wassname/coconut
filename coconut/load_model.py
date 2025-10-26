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
    else:
        quantization_config = None  # Default no quant for TRM if not specified

    base_model = AutoModelForCausalLM.from_pretrained(
        conf.model_id, config=model_config, device_map=device, torch_dtype=dtype, quantization_config=quantization_config
    )

    base_model.resize_token_embeddings(len(tokenizer))

    logger.debug("Loading TRM LoRA adapter")

    num_layers = base_model.config.num_hidden_layers
    # trm_hidden_size = base_model.config.hidden_size


    target_layers = torch.linspace(int(num_layers*conf.layers_start_adapter), int(num_layers*conf.layers_end_adapter), steps=conf.layers_spacing_adapter).long().tolist()
    target_layers = sorted(set(target_layers))

    logger.info(f"Targeting LoRA layers: {target_layers} out of {num_layers} total layers")
    target_modules = [k for k,v in base_model.named_modules() if  isinstance(v, torch.nn.Linear) and any(f".{i}." in k for i in target_layers)]
    logger.debug(f"Targeting {len(target_modules)} modules for TRM LoRA adapters: {target_modules}")


    adapter_config_args = {}
    prefixes = ['adapter_', ]
    for k in conf.__dataclass_fields__.keys():
        for p in prefixes:
            if k.startswith(p):
                kk = k[len(p):]
                adapter_config_args[kk] = getattr(conf, k)
                break

    logger.debug(f"Adapter config args: {adapter_config_args}")
    
    AdapterConfig = conf._adapter_class
    peft_config = AdapterConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        # r=conf.lora_r,
        # lora_alpha=conf.lora_alpha,
        # lora_dropout=conf.lora_dropout,
        # target_modules="all-linear",  # Target all linear layers
        target_modules=target_modules,
        bias="none",

        l_cycles=conf.trm_l_cycles,
        h_cycles=conf.trm_h_cycles,
        expansion=conf.trm_expansion,
        l_layers=conf.trm_l_layers,
        num_heads=conf.trm_num_heads,
        # update_mode='lora',
        modules_to_save=None,
        **adapter_config_args
    )
    # Use TRMLoraModel directly instead of get_peft_model
    # peft_model = TRMLoraModel(base_model, peft_config, "default")

    peft_model = get_peft_model(base_model, peft_config)
    logger.debug("Completed loading TRM LoRA adapter")
    peft_model.enable_input_require_grads()

    peft_model.print_trainable_parameters()

    assert peft_model.base_model.model.lm_head.weight.requires_grad is False, "LM head weights should be frozen"
    assert peft_model.base_model.model.model.embed_tokens.weight.requires_grad is False, "Embedding weights should be frozen"

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
    # logger.error("FIXME save with custom peft type: Unknown PEFT type passed: TRMLORA")
    # need to add to PEFT_TYPE_TO_PREFIX_MAPPING
    # https://github.com/huggingface/peft/blob/98a88c01a42be4bb2fa13a1c0dd5340c42f82c87/src/peft/utils/save_and_load.py#L228
    model.model.save_pretrained(save_folder, save_embedding_layers=False)
    logger.info(f"saving model {save_folder}")
