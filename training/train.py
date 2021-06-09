import os
import sys
from tensorflow.keras import callbacks as KC

sys.path.append("..")
import utils
from .model import define_model
from utils import DataGenerator


def train(args):
    # Read config
    config = utils.get_config(args.config)

    # Create logdir
    log_dir = config["training"]["log_dir"]
    # Create weights dir
    weights_dir = os.path.join(log_dir, "weights")
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    # Create file contains labels name
    label_file = os.path.join(log_dir, config["training"]["labels"])
    with open(label_file, "w") as fo:
        pass

    # Load dataset
    # Labels name will be written to file in train_generator
    train_generator = DataGenerator(config, subset="train_set")
    val_generator = DataGenerator(config, subset="val_set")

    # Define model
    input_shape = train_generator.get_input_shape()
    num_classes = train_generator.get_num_classes()
    net = define_model(input_shape, num_classes)
    net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    net.summary()

    # Define callbacks
    csv_callback = KC.CSVLogger(filename=os.path.join(log_dir, "log.csv"))
    tensorboard = KC.TensorBoard(
        log_dir=os.path.join(log_dir, "log"),
        update_freq=config["training"]["update_freq"],
    )
    model_ckpt = KC.ModelCheckpoint(
        filepath=os.path.join(weights_dir, "{epoch:04d}-{val_loss:.4f}.hdf5")
    )
    net.fit_generator(
        generator=train_generator,
        steps_per_epoch=config["training"]["train_steps"],
        epochs=config["training"]["epochs"],
        callbacks=[csv_callback, tensorboard, model_ckpt],
        validation_data=val_generator,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    args = parser.parse_args()
    train(args)
