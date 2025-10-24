from coconut.configs import BaseConfig, TRMLoRADebug
from coconut.train import train
import coconut.silence

def test_tiny_train():
    # should train a small amount, eval, save
    conf = TRMLoRADebug()
    train(conf)

# TODO test forward
# model, tokenizer = load_new_model(conf, device, dtype)



# TODO move run to library, so we can do a quick test with debug config
