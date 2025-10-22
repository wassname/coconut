import torch
from coconut.load_model import load_new_model
from coconut.configs import BaseConfig
from loguru import logger
from transformers import AutoTokenizer
import yaml

# Simple config for test
conf_dict = {
    'model_id': 'Qwen/Qwen2.5-Coder-0.5B',
    'use_trm_lora': True,
    'load_in_4bit': True,
    'use_position_ids': False,
    'pad_token_id': None,
    'eos_token_id': None,
}
conf = BaseConfig(**conf_dict)
logger.info(f"Config: {conf}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

model, tokenizer = load_new_model(conf, device, dtype)

# Dummy input
input_ids = tokenizer("What is 2 + 2?", return_tensors='pt').input_ids.to(device)
attention_mask = torch.ones_like(input_ids)

logger.info("Model loaded, testing forward with adapter")

# Forward with adapter enabled
with torch.no_grad():
    outputs_enabled = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

hidden_enabled_layer20 = outputs_enabled.hidden_states[20].mean().item()
logger.info(f"Layer 20 hidden mean (enabled): {hidden_enabled_layer20}")

# Disable adapter and forward again
if hasattr(model.model, 'disable_adapters'):
    model.model.disable_adapters()
    with torch.no_grad():
        outputs_disabled = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    hidden_disabled_layer20 = outputs_disabled.hidden_states[20].mean().item()
    logger.info(f"Layer 20 hidden mean (disabled): {hidden_disabled_layer20}")
    
    diff = abs(hidden_enabled_layer20 - hidden_disabled_layer20)
    logger.info(f"Difference in layer 20: {diff}")
    if diff > 1e-6:
        logger.info("SUCCESS: Hiddens modified by TRM LoRA adapter!")
    else:
        logger.info("WARNING: No modification detected; check adapter targeting.")

# Re-enable if needed
if hasattr(model.model, 'enable_adapters'):
    model.model.enable_adapters()

# Quick generation test
logger.info("Testing generation")
generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
logger.info(f"Generated: {generated_text}")

logger.info("Test completed.")
