
"""
see also
- https://github.com/JD-P/minihf/blob/9e64b1ffb44c00ebab933301a80b902f422faba4/minihf_infer.py#L37
 https://github.com/wassname/open_pref_eval/blob/c5aaa23d90ae0c61cfaab0a8ab40ea0ea1cdae64/open_pref_eval/helpers/peft_utils.py#L41
"""

from contextlib import contextmanager, nullcontext
from transformers.utils import is_peft_available
from transformers import PreTrainedModel
from loguru import logger
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from contextlib import contextmanager
from peft import PeftModel
from typing import Optional

if is_peft_available():
    from peft import AutoPeftModelForCausalLM, get_peft_model, PeftConfig, PeftModelForCausalLM
    from peft import PeftModel, get_peft_model
    

def is_peft_model(model):
    return is_hf_peft_model(model) or is_plain_peft_model(model)

def is_plain_peft_model(model):
    if is_peft_available() and isinstance(model, PeftModel):
        return True
    return False

def is_hf_peft_model(model):
    if is_peft_available() and hasattr(model, 'peft_config'):
        return True
    return False


@contextmanager
def set_adapter(model, adapter_name: str = None):
    """
    Context manager to set the adapter for a model.
    If adapter_name is None, it disables the adapters.
    """
    if is_plain_peft_model(model):
        yield from set_peft_adapter(model, adapter_name)
    elif is_hf_peft_model(model):
        yield from set_hf_adapter(model, adapter_name)
    else:
        raise ValueError("Model is not a PEFT model or HF PEFT model.")

# @contextmanager
def set_hf_adapter(model: PreTrainedModel, adapter_name: str = None):
    old_adapter_name = model.active_adapter()
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            model.disable_adapters()
            yield model
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise e
    finally:
        if old_adapter_name is None:
            model.disable_adapters()
        else:
            model.enable_adapters()
            model.set_adapter(old_adapter_name)

# @contextmanager
def set_peft_adapter(model: PeftModel, adapter_name: str = None):
    old_adapter_name = model.active_adapter
    try:
        if adapter_name is not None:
            model.set_adapter(adapter_name)
            yield model
        else:
            with model.disable_adapter():
                yield model
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise e
    finally:
        if old_adapter_name is None:
            model.disable_adapter()
        else:
            model.set_adapter(old_adapter_name)


# @contextmanager
# def set_adapter(model: PeftModel, adapter_name: Optional[str] = None):
#     old_adapter_name = model.active_adapter
#     try:
#         if adapter_name is not None:
#             model.set_adapter(adapter_name)
#             yield model
#         else:
#             with model.disable_adapter():
#                 yield model
#     finally:
#         model.set_adapter(old_adapter_name)
