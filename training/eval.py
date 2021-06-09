import os
import sys
from tensorflow.keras.models import load_model

sys.path.append("..")
import utils
from utils import DataGenerator


def eval(args):
    # Read config
    config = utils.get_config(args.config)

    # Load dataset
    # Labels name will be written to file in train_generator
    eval_generator = DataGenerator(config, subset="test_set")

    # Load model
    net = load_model(args.pretrained)
    net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    net.summary()

    # Define callbacks
    net.evaluate_generator(generator=eval_generator, verbose=1)
    # Current: loss: 0.1530 - acc: 0.9526


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--pretrained", type=str, help="Pretrained weights")
    args = parser.parse_args()
    eval(args)
