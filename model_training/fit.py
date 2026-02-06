"""
This script fits a NN model using clipnet. It requires a NN architecture file, which
must contain a function named construct_nn that returns a tf.keras.models.Model object.
It also requires a dataset_params.json file which specifies parameters and file paths
associated with the dataset of interest.
"""

import argparse
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf
from clipnet import clipnet


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="directory to save models to. Must contain a dataset_params.json file.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="name of model to save."
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="resume training from this model.",
    )
    args = parser.parse_args()

    if len(tf.config.list_physical_devices("GPU")) > 0:
        nn = clipnet.CLIPNET(name=args.name, n_gpus=1, use_specific_gpu=0)
    else:
        nn = clipnet.CLIPNET(name=args.name, n_gpus=0)
    nn.fit(model_dir=args.model_dir, resume_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()
