
# replace ENUM with extended version, we need to replace
import peft.utils.peft_types
import enum
from peft.utils import register_peft_method
from coconut.trmlora.recursive_lora import TRMLoraModel, TRMConfig

class PeftType2(str, enum.Enum):
    TRMLORA = 'TRMLORA'
peft.utils.peft_types.PeftType = PeftType2

try:
    register_peft_method(name="trmlora", model_cls=TRMLoraModel, config_cls=TRMConfig)
except KeyError:
    # already registered
    pass
