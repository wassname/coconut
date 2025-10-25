# replace ENUM with extended version, we need to replace
import peft.utils.peft_types
import enum
from peft.utils import register_peft_method
from coconut.trmlora.recursive_lora import TRMLoraModel, TRMConfig
from coconut.trmlora.recursive_delora import TRMDeloraModel, TRMDeloraConfig
from coconut.trmlora.recursive_hra import TRMHraModel, TRMHraConfig

class PeftType2(str, enum.Enum):
    TRMLORA = 'TRMLORA'
    TRMDELORA = 'TRMDELORA'
    TRMHRA = 'TRMHRA'
peft.utils.peft_types.PeftType = PeftType2

try:
    register_peft_method(name="trmlora", model_cls=TRMLoraModel, config_cls=TRMConfig, prefix="lora_")
except KeyError:
    # already registered
    pass

try:
    register_peft_method(name="trmdelora", model_cls=TRMDeloraModel, config_cls=TRMDeloraConfig, prefix="delora_")
except KeyError:
    # already registered
    pass

try:
    register_peft_method(name="trmhra", model_cls=TRMHraModel, config_cls=TRMHraConfig, prefix="hra_")
except KeyError:
    # already registered
    pass


from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
PEFT_TYPE_TO_PREFIX_MAPPING[PeftType2.TRMLORA] = "lora_"
PEFT_TYPE_TO_PREFIX_MAPPING[PeftType2.TRMDELORA] = "delora_"
PEFT_TYPE_TO_PREFIX_MAPPING[PeftType2.TRMHRA] = "hra_"
