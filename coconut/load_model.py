from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)
from coconut.coconut import (
    CoconutConfig,
    CoconutQwen3ForCausalLM,
)
from coconut.configs import BaseConfig
from loguru import logger
from pathlib import Path
import torch
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

    # load base model
    model_config = CoconutConfig.from_pretrained(
        conf.model_id,
        latent_token_id=latent_id,
        bot_id=bot_id,
        eot_id=eot_id,
        eos_token_id=tokenizer.eos_token_id,
        use_position_ids=conf.use_position_ids,
        replacement_method=conf.replacement_method,
        loss_seq_vcr=conf.loss_seq_vcr,
        n_detached_recursions=conf.n_detached_recursions,
        use_trm=getattr(conf, 'use_trm', False),
        load_in_4bit=getattr(conf, 'load_in_4bit', False),
        load_in_8bit=getattr(conf, 'load_in_8bit', False),
        trm_num_layers=getattr(conf, 'trm_num_layers', 2),
        trm_num_heads=getattr(conf, 'trm_num_heads', 8),
        trm_expansion=getattr(conf, 'trm_expansion', 2.67),
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
        
        model = CoconutQwen3ForCausalLM.from_pretrained(
            conf.model_id, 
            config=model_config, 
            device_map=device, 
            torch_dtype=dtype,
            quantization_config=quantization_config
        )

    model = CoconutQwen3ForCausalLM.from_pretrained(
        conf.model_id, config=model_config, device_map=device, torch_dtype=dtype, quantization_config=quantization_config
    )
    
    # apply_config(model, tokenizer, conf)

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def resume_model(conf: BaseConfig, device="auto", dtype=torch.bfloat16):
    # load model
    f = Path("./" + conf.load_model_path)
    assert f.exists(), f"Model path {f} does not exist"
    tokenizer = AutoTokenizer.from_pretrained(conf.load_model_path)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    config = CoconutConfig.from_pretrained(
        conf.load_model_path,
        latent_token_id=latent_id,
        bot_id=bot_id,
        eot_id=eot_id,
        eos_token_id=tokenizer.eos_token_id,
        use_position_ids=conf.use_position_ids,
        loss_seq_vcr=conf.loss_seq_vcr,
        replacement_method=conf.replacement_method,
        n_detached_recursions=conf.n_detached_recursions,
        load_in_4bit=conf.load_in_4bit,
        load_in_8bit=conf.load_in_8bit,
        use_trm=conf.use_trm,
        trm_num_layers=conf.trm_num_layers,
        trm_num_heads=conf.trm_num_heads,
        trm_expansion=conf.trm_expansion,
        # need
    )


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
        
    model = CoconutQwen3ForCausalLM.from_pretrained(
        conf.load_model_path,
        config=config,
        # conf.load_model_path, 
        device_map=device, torch_dtype=dtype, quantization_config=quantization_config
    )
    logger.warning(f"Resumed model from {conf.load_model_path}")

    # apply_config(model, tokenizer, conf)

    # set the configuration
    return model, tokenizer


# def apply_config(model, tokenizer, conf: Config):
#     model.config.latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
#     model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
#     model.config.use_position_ids = conf.use_position_ids
#     model.config.loss_seq_vcr = conf.loss_seq_vcr
#     model.config.replacement_method = conf.replacement_method

def tie_embeddings(model, tokenizer):
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    bot_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    # tie the embeddings for the special tokens
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
    return model

def save_model(model, tokenizer, configs, save_dir: Path):
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    with open(save_dir / "coconut_config.toml", "w") as f:
        toml.dump(configs, f)
    logger.info(f"saving model {save_dir}")
