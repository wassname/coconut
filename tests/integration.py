from coconut.configs import BaseConfig, Debug
from coconut.train import train


def tiny_train():
    conf = Debug()
    train(conf)

# TODO test forward
# model, tokenizer = load_new_model(conf, device, dtype)



# TODO move run to library, so we can do a quick test with debug config
