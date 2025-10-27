# replace ENUM with extended version, we need to replace
import enum

# Monkey patching to add new PEFT types
import peft.utils.peft_types
class PeftType2(str, enum.Enum):
    TRMLORA = 'TRMLORA'
    TRMDELORA = 'TRMDELORA'
    TRMHRA = 'TRMHRA'
    TRMSVFT = 'TRMSVFT'
peft.utils.peft_types.PeftType = PeftType2


from peft.utils import register_peft_method

from peft.utils.save_and_load import get_peft_model_state_dict as orig_get_peft_model_state_dict
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
from peft.utils.peft_types import PeftType
import re

"""Monkey patch to support custom PEFT types in get_peft_model_state_dict."""
def custom_get_peft_model_state_dict(*args, **kwargs):
    try:
        return orig_get_peft_model_state_dict(*args, **kwargs)
    except ValueError as e:
        if "Unknown PEFT type passed" not in str(e):
            raise
        
        # Handle custom PEFT types
        model = args[0]
        adapter_name = kwargs.get('adapter_name', 'default')
        config = model.peft_config[adapter_name]
        
        print(f"[TRM PATCH] Handling custom PEFT type: {config.peft_type}")
        
        if config.peft_type not in PEFT_TYPE_TO_PREFIX_MAPPING:
            raise
        
        state_dict = kwargs.get('state_dict')
        if state_dict is None:
            state_dict = model.state_dict()
        
        prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
        to_return = {k: state_dict[k] for k in state_dict if prefix in k}
        
        # Remove adapter name from keys (same logic as original function)
        pattern = re.compile(re.escape(f".{adapter_name}") + r"$")
        
        def remove_adapter_name(key):
            if "." not in key:
                return key
            if key.endswith(f".{adapter_name}"):
                return key.removesuffix(f".{adapter_name}")
            key, _, suffix = key.rpartition(".")
            key = pattern.sub("", key)
            return f"{key}.{suffix}"
        
        to_return = {remove_adapter_name(k): v for k, v in to_return.items()}
        return to_return

import peft.utils.save_and_load
import peft.utils
peft.utils.save_and_load.get_peft_model_state_dict = custom_get_peft_model_state_dict
peft.utils.get_peft_model_state_dict = custom_get_peft_model_state_dict  # peft_model.py imports from here


from coconut.trmlora.recursive_lora import TRMLoraModel, TRMLoraAConfig
from coconut.trmlora.recursive_delora import TRMDeloraModel, TRMDeloraAConfig
from coconut.trmlora.recursive_hra import TRMHraModel, TRMHraAConfig
from coconut.trmlora.recursive_svft import TRMSvftModel, TRMSvftConfig



try:
    register_peft_method(name="trmlora", model_cls=TRMLoraModel, config_cls=TRMLoraAConfig, prefix="lora_")
except KeyError:
    pass

try:
    register_peft_method(name="trmdelora", model_cls=TRMDeloraModel, config_cls=TRMDeloraAConfig, prefix="delora_")
except KeyError:
    pass

try:
    register_peft_method(name="trmhra", model_cls=TRMHraModel, config_cls=TRMHraAConfig, prefix="hra_")
except KeyError:
    pass

try:
    register_peft_method(name="trmsvft", model_cls=TRMSvftModel, config_cls=TRMSvftConfig, prefix="svft_")
except KeyError:
    pass

PEFT_TYPE_TO_PREFIX_MAPPING[PeftType2.TRMLORA] = "lora_"
PEFT_TYPE_TO_PREFIX_MAPPING[PeftType2.TRMDELORA] = "delora_"
PEFT_TYPE_TO_PREFIX_MAPPING[PeftType2.TRMHRA] = "hra_"
PEFT_TYPE_TO_PREFIX_MAPPING[PeftType2.TRMSVFT] = "svft_"
