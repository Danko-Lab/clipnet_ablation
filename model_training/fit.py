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
import clipnet_all_checkpoints as clipnet


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
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=2,
        help=(
            "Number of GPUs used for distributed training. The published-model "
            "configuration uses 2 GPUs. Use 0 for CPU training."
        ),
    )
    args = parser.parse_args()

    if args.n_gpus < 0:
        parser.error("--n_gpus must be 0 or a positive integer.")

    available_gpus = len(tf.config.list_physical_devices("GPU"))
    if args.n_gpus > available_gpus:
        parser.error(
            f"--n_gpus {args.n_gpus} requested, but TensorFlow sees "
            f"{available_gpus} GPU(s)."
        )

    if args.n_gpus > 0:
        nn = clipnet.CLIPNET(
            name=args.name,
            n_gpus=args.n_gpus,
            use_specific_gpu=0 if args.n_gpus == 1 else None,
        )
    else:
        nn = clipnet.CLIPNET(name=args.name, n_gpus=0, use_specific_gpu=-1)
    nn.fit(model_dir=args.model_dir, resume_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()
