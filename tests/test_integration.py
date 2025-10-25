import pytest
import coconut.silence
from coconut.configs import BaseConfig, TRMLoRADebug, TRMDeloraDebug, TRMHraDebug
from coconut.train import train

@pytest.mark.parametrize(
    "config_class",
    [
        TRMLoRADebug,
        TRMDeloraDebug,
        TRMHraDebug,
    ],
)

def test_tiny_train(config_class):
    # should train a small amount, eval, save
    conf = config_class()
    train(conf)

# TODO test forward
# model, tokenizer = load_new_model(conf, device, dtype)



# TODO move run to library, so we can do a quick test with debug config
