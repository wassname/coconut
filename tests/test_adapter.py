import torch
import warnings
import pytest
from peft import get_peft_model, LoraConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut.adapters import is_hf_peft_model, is_plain_peft_model
from coconut.trmlora.recursive_lora import TRMConfig, TRMLoraModel
from coconut.trmlora.recursive_delora import TRMDeloraConfig, TRMDeloraModel
from coconut.gen import gen, gen_sample

# Silence the pydantic warnings from tyro configs
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal")
warnings.filterwarnings("ignore", message="UnsupportedFieldAttributeWarning")


@pytest.mark.parametrize(
    "config_class, expected_model_class, adapter_name",
    [
        (LoraConfig, PeftModel, "lora"),
        (TRMConfig, TRMLoraModel, "trmlora"),
        (TRMDeloraConfig, TRMDeloraModel, "trmdelora"),
    ],
)
def test_adapter(config_class, expected_model_class, adapter_name):
    # load a tiny model 
    model_id = "yujiepan/qwen3-tiny-random"
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    peft_config = config_class(r=4)  # Low rank for quick test
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Assert it's a PEFT model
    assert is_hf_peft_model(model) or is_plain_peft_model(model)
    
    # Assert correct subclass for custom adapters
    assert isinstance(model, PeftModel)

    # randomize adapter weights for test
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.1)

    # Test forward pass with dummy input
    s1 = gen_sample(model, tokenizer)

    # now make sure disabling works?
    with model.disable_adapter():
        s2 = gen_sample(model, tokenizer)

    # make sure they are different
    assert s1 != s2

    # test save
    save_path = f"/tmp/qwen-tiny-{adapter_name}"
    model.save_pretrained(save_path)

    # Test load
    loaded_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_id), save_path)
    assert isinstance(loaded_model, PeftModel)
    s3 = gen_sample(loaded_model, tokenizer)

