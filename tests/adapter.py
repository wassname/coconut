from coconut.adapters import set_adapter, is_hf_peft_model, is_plain_peft_model
from coconut.recursive_lora import TRMConfig, TRMLoraLayer, TRMModel



def test_set_adapter():
    # load a tiny model 
    model_id = "yujiepan/qwen3-tiny-random"
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    peft_config = LoraConfig()
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()



    assert is_hf_peft_model(model) or is_plain_peft_model(model)

    model.save_pretrained("/tmp/qwen2.5-3b-lora")
