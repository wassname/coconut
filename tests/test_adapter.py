import coconut.trmlora  # ensure trmlora adapters are registered
import coconut.silence
import torch
import torch.nn as nn
import warnings
import pytest
from pathlib import Path
from peft import get_peft_model, LoraConfig
from peft import PeftModel
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut.adapters import is_hf_peft_model, is_plain_peft_model
from coconut.trmlora.recursive_lora import TRMLoraAConfig, TRMLoraModel
from coconut.trmlora.recursive_delora import TRMDeloraAConfig, TRMDeloraModel
from coconut.trmlora.recursive_hra import TRMHraAConfig, TRMHraModel
from coconut.trmlora.recursive_svft import TRMSvftAConfig, TRMSvftModel
from coconut.gen import gen, gen_sample
from coconut.load_model import Coconut, load_new_model, save_model, load_adapter


@pytest.mark.parametrize(
    "config_class, expected_model_class, adapter_name",
    [
        (LoraConfig, PeftModel, "lora"),
        (TRMLoraAConfig, TRMLoraModel, "trmlora"),
        (TRMDeloraAConfig, TRMDeloraModel, "trmdelora"),
        (TRMHraAConfig, TRMHraModel, "trmhra"),
        (TRMSvftAConfig, TRMSvftModel, "trmsvft"),
    ],
)
def test_adapter(config_class, expected_model_class, adapter_name):
    # load a tiny model 
    model_id = "yujiepan/qwen3-tiny-random"
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    peft_config = config_class()  # Low rank for quick test

    # Make using method 1
    # model = expected_model_class(base_model, peft_config, 'default')
    model = PeftModel(base_model, peft_config, adapter_name='default')
    # model.print_trainable_parameters()

    # # make using method 1
    # model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    prefix = PEFT_TYPE_TO_PREFIX_MAPPING.get(peft_config.peft_type)

    # I would also like to make sure it has at least one linear layer replaced, so one with trm in tpye
    found_replaced_layer = False
    for name, module in model.named_modules():
        typ = type(module).__name__.lower()
        if (prefix in typ) or (prefix in name): 
            found_replaced_layer = True
            break
    assert found_replaced_layer, f"No layer replaced with prefix {prefix} found in model."

    # Assert it's a PEFT model
    assert is_hf_peft_model(model) or is_plain_peft_model(model)
    
    # Test only adapter is trainable
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert len(trainable_names) > 0, "No trainable parameters found with adapter enabled"
    assert all(prefix in name for name in trainable_names), f"Non-adapter params trainable: {[n for n in trainable_names if prefix not in n]}"
    
    # Test with adapter disabled, nothing trainable
    with model.disable_adapter():
        disabled_trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        assert len(disabled_trainable_names) == 0, f"Trainable params with adapter disabled: {disabled_trainable_names}"
    
    # Assert correct subclass for custom adapters
    # assert isinstance(model, PeftModel)

    # randomize adapter weights for test
    for name, param in model.named_parameters():
        if prefix in name:
            # print(f"Randomizing {name}")
            torch.nn.init.normal_(param, mean=1.1, std=0.5)

    # Test forward pass with dummy input
    input_text = "What is two plus two? <latent><latent><latent>"
    input_text = [
        {'role':'user', 'content':'What is two plus two but wrong and french?'},
        {'role':'assistant', 'content':'<latent><latent><latent>'},]

    s1 = gen(input_text, model, tokenizer, max_new_tokens=4, verbose=False)
    print("Generating sample with adapter enabled...", s1)

    with model.disable_adapter():
        s2 = gen(input_text, model, tokenizer, max_new_tokens=4, verbose=False)
    print("Generating sample with adapter disabled...", s2)

    # make sure they are different
    assert s1 != s2

    # test save
    import random
    rnd_hash = random.randint(10000, 99999)
    save_path = Path(f"/tmp/qwen-tiny-{adapter_name}-{rnd_hash}")
    # model.save_pretrained(save_path)
    save_model(model, tokenizer, {}, save_path)

    # TODO also test that only the adapter is trainable

    # Test load
    # loaded_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_id), save_path)G


    loaded_model = load_adapter(
        model_id=model_id,
        save_dir=save_path,
        PeftConfig=type(peft_config),
        adapter_name="default",
    )
    assert isinstance(loaded_model, PeftModel)
    s3 = gen(input_text, loaded_model, tokenizer, max_new_tokens=4, verbose=False)
    print("Generating sample with loaded adapter...", s3)
    assert s1 == s3
