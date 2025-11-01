from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from coconut.coconut import Coconut
from coconut.configs import BaseConfig, TRMSvft
from loguru import logger
from pathlib import Path

import torch
import safetensors.torch
import toml
from typing import Optional
from transformers import BitsAndBytesConfig
from coconut.trmlora import PEFT_TYPE_TO_PREFIX_MAPPING
from peft.utils.save_and_load import _insert_adapter_name_into_state_dict
from peft import PeftModel, PeftConfig
from coconut import trmlora  # ensure trmlora is imported to register peft types
# from coconut.configs import bot_token, latent_token, eot_token


def coconut_to_adapter_config_converter(conf: BaseConfig, target_modules: Optional[list[str]] = None) -> PeftConfig:
    adapter_config_args = {}
    prefixes = [
        "adapter_",
    ]
    for k in conf.__dataclass_fields__.keys():
        for p in prefixes:
            if k.startswith(p):
                kk = k[len(p) :]
                adapter_config_args[kk] = getattr(conf, k)
                break

    AdapterConfig = conf._adapter_class
    peft_config = AdapterConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        # r=conf.lora_r,
        # lora_alpha=conf.lora_alpha,
        # lora_dropout=conf.lora_dropout,
        # target_modules="all-linear",  # Target all linear layers
        target_modules=target_modules,
        # bias="none",
        l_cycles=conf.trm_l_cycles,
        h_cycles=conf.trm_h_cycles,
        expansion=conf.trm_expansion,
        l_layers=conf.trm_l_layers,
        num_heads=conf.trm_num_heads,
        # update_mode='lora',
        modules_to_save=None,
        **adapter_config_args,
    )
    return peft_config


def load_new_model(conf: BaseConfig, device, dtype):
    latent_token  = conf.latent_token
    bot_token     = conf.bot_token
    eot_token     = conf.eot_token


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        conf.model_id,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.convert_tokens_to_ids(latent_token) is None:
        raise ValueError(f"Latent token `{latent_token}` not in tokenizer vocab, please chose a single in sample token.")
        tokenizer.add_tokens(latent_token)
    if tokenizer.convert_tokens_to_ids(bot_token) is None:
        raise ValueError(f"Bot token `{bot_token}` not in tokenizer vocab, please chose a single in sample token.")
        tokenizer.add_tokens(bot_token)
    if tokenizer.convert_tokens_to_ids(eot_token) is None:
        raise ValueError(f"Eot token `{eot_token}` not in tokenizer vocab, please chose a single in sample token.")
        tokenizer.add_tokens(eot_token)

    latent_id = tokenizer.convert_tokens_to_ids(latent_token)
    bot_id = tokenizer.convert_tokens_to_ids(bot_token)
    eot_id = tokenizer.convert_tokens_to_ids(eot_token)

    conf.latent_token = latent_token
    conf.bot_token = bot_token
    conf.eot_token = eot_token
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
    if getattr(conf, "load_in_4bit", False):
        logger.info("Loading in 4bit")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif getattr(conf, "load_in_8bit", False):
        logger.info("Loading in 8bit")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None  # Default no quant for TRM if not specified

    base_model = AutoModelForCausalLM.from_pretrained(
        conf.model_id,
        config=model_config,
        device_map=device,
        torch_dtype=dtype,
        quantization_config=quantization_config,
    )

    base_model.resize_token_embeddings(len(tokenizer))

    logger.debug("Loading TRM LoRA adapter")

    num_layers = base_model.config.num_hidden_layers
    # trm_hidden_size = base_model.config.hidden_size

    target_layers = (
        torch.linspace(
            int(num_layers * conf.layers_start_adapter),
            int(num_layers * conf.layers_end_adapter),
            steps=conf.layers_spacing_adapter,
        )
        .long()
        .tolist()
    )
    target_layers = sorted(set(target_layers))

    logger.info(
        f"Targeting LoRA layers: {target_layers} out of {num_layers} total layers"
    )
    target_modules = [
        k
        for k, v in base_model.named_modules()
        if isinstance(v, torch.nn.Linear) and any(f".{i}." in k for i in target_layers)
    ]

    if conf.target_modules_pattern:
        import re
        print('target_modules', target_modules[:10], '...')
        target_modules = [m for m in target_modules if re.search(conf.target_modules_pattern, m)]
        logger.info(f"Filtering target modules with pattern '{conf.target_modules_pattern}': {target_modules}")
        assert len(target_modules) > 0, "No target modules matched the given pattern!"
    logger.debug(
        f"Targeting {len(target_modules)} modules for TRM LoRA adapters: {target_modules}"
    )
    # FIXME is it missing k_proj?

    adapter_config_args = {}
    prefixes = [
        "adapter_",
    ]
    for k in conf.__dataclass_fields__.keys():
        for p in prefixes:
            if k.startswith(p):
                kk = k[len(p) :]
                adapter_config_args[kk] = getattr(conf, k)
                break

    logger.debug(f"Adapter config args: {adapter_config_args}")

    peft_config = coconut_to_adapter_config_converter(conf, target_modules=target_modules)

    # AdapterConfig = conf._adapter_class
    # peft_config = AdapterConfig(
    #     task_type="CAUSAL_LM",
    #     inference_mode=False,
    #     # r=conf.lora_r,
    #     # lora_alpha=conf.lora_alpha,
    #     # lora_dropout=conf.lora_dropout,
    #     # target_modules="all-linear",  # Target all linear layers
    #     target_modules=target_modules,
    #     # bias="none",
    #     l_cycles=conf.trm_l_cycles,
    #     h_cycles=conf.trm_h_cycles,
    #     expansion=conf.trm_expansion,
    #     l_layers=conf.trm_l_layers,
    #     num_heads=conf.trm_num_heads,
    #     # update_mode='lora',
    #     modules_to_save=None,
    #     **adapter_config_args,
    # )
    # Use TRMLoraModel directly instead of get_peft_model
    # peft_model = TRMLoraModel(base_model, peft_config, "default")

    # peft_model = get_peft_model(base_model, peft_config)
    peft_model = PeftModel(base_model, peft_config, adapter_name='default')
    logger.debug("Completed loading TRM LoRA adapter")
    peft_model.enable_input_require_grads()

    peft_model.print_trainable_parameters()

    assert peft_model.base_model.model.lm_head.weight.requires_grad is False, (
        "LM head weights should be frozen"
    )
    assert (
        peft_model.base_model.model.model.embed_tokens.weight.requires_grad is False
    ), "Embedding weights should be frozen"



    model = Coconut(peft_model, conf)
    # if it's svft all base weights should be cpu?
    if isinstance(conf, trmlora.TRMSvftAConfig):
        logger.info("Moving base model weights to CPU for SVFT")
        for name, p in model.model.base_model.named_parameters():
            if 'base_layer' in name:
                p.data = p.data.to("cpu")
        # model.model.base_model.base_model.base_model.to("cpu")
        # example check
        #
    # model.model.base_model.base_model.layers[23].self_attn.q_proj.base_layer.weight
    return model, tokenizer


def resume_model(conf: BaseConfig, device="auto", dtype=torch.bfloat16):
    model, tokenizer = load_new_model(conf, device, dtype)

    if getattr(conf, "use_trm_lora", False):
        logger.info("Loading TRM LoRA adapter weights")
        model.model.load_adapter(conf.load_model_path)
        logger.info(f"Resumed TRM LoRA adapter from {conf.load_model_path}")
    else:
        state_dict = safetensors.torch.load_file(conf.load_model_path, device=device)
        model.load_state_dict(state_dict, strict=False)
        logger.warning(f"Resumed model from {conf.load_model_path}")

    # set the configuration
    return model, tokenizer


def save_model(model, tokenizer, configs, save_dir: Path, adapter_name="default"):
    """Peft is to hard to subclass or monkey patch, in the end I needed by own function."""
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "coconut_config.toml", "w") as f:
        toml.dump(configs, f)
    logger.info(f"saving model {save_dir}")

    # save state dict (only TRM adapter, not frozen base model)
    save_folder = save_dir / "trmlora/"
    save_folder.mkdir(parents=True, exist_ok=True)


    config = model.model.peft_config[adapter_name]
    state_dict = model.state_dict()

    # Filter by prefix (same logic as PEFT but without type check)
    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
    to_return = {k: state_dict[k] for k in state_dict if prefix in k}

    # Remove adapter name from keys
    def remove_adapter_name(key):
        if "." not in key:
            return key
        if key.endswith(f".{adapter_name}"):
            return key.removesuffix(f".{adapter_name}")
        return key.replace(f".{adapter_name}.", ".")

    to_return = {remove_adapter_name(k): v for k, v in to_return.items()}

    assert not any(adapter_name in k for k in to_return.keys()), "Adapter name still present in saved keys"

    # Save adapter weights
    # torch.save(to_return, os.path.join(save_folder, "adapter_model.bin"))
    safetensors.torch.save_file(
        to_return,
        save_folder/ "adapter_model.safetensors",
    )

    # Save adapter config
    config.save_pretrained(save_folder)

    logger.info(f"Saved TRM adapter to {save_folder}")


def load_adapter(
    model_id: str,
    Config: BaseConfig,
    save_dir: Path,
    adapter_name="default",
    torch_device="cuda",
    autocast_adapter_dtype: bool = True,
    ephemeral_gpu_offload: bool = False,
    low_cpu_mem_usage: bool = False,
    key_mapping: Optional[dict[str, str]] = None,
):
    """Peft is to hard to subclass or monkey patch, in the end I needed by own function."""
    # TODO tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(model_id)

    f = Path(save_dir) / 'coconut_config.toml'
    import tomli
    with open(f, 'rb') as fp:
        conf_dict = tomli.load(fp)
    conf = Config(**conf_dict,)
    peft_config = coconut_to_adapter_config_converter(conf)

    peft_model = PeftModel(
        base_model,
        peft_config,
        adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    # LOAD coconut...
    model = Coconut(peft_model, conf)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


    adapter_save_path = save_dir / "trmlora/"
    state_dict = safetensors.torch.load_file(adapter_save_path/ "adapter_model.safetensors", device=torch_device)

    assert not any(adapter_name in k for k in state_dict.keys()), "Adapter name should not be present in loaded keys"

    parameter_prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_config.peft_type]
    peft_model_state_dict = _insert_adapter_name_into_state_dict(
        state_dict, adapter_name=adapter_name, parameter_prefix=parameter_prefix
    )

    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    logger.warning(f"Loading adapter weights from {model_id} resulted in {load_result}")
    return tokenizer, model
