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
        "--epochs",
        type=int,
        default=None,
        help="number of epochs to train in this call (default: clipnet_arch.epochs).",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=0,
        help=(
            "epoch the resumed checkpoint ended on; offsets checkpoint names and "
            "the training log."
        ),
    )
    parser.add_argument(
        "--restore_optimizer",
        action="store_true",
        help="restore the optimizer state saved in --resume_checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=None,
        help="retain a checkpoint every N epochs (default: clipnet_arch.checkpoint_frequency).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for Python, NumPy, and TensorFlow (data order and dropout).",
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
    if args.restore_optimizer and args.resume_checkpoint is None:
        parser.error("--restore_optimizer requires --resume_checkpoint.")
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)

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
    nn.fit(
        model_dir=args.model_dir,
        resume_checkpoint=args.resume_checkpoint,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        restore_optimizer=args.restore_optimizer,
        checkpoint_frequency=args.checkpoint_frequency,
    )


if __name__ == "__main__":
    main()
