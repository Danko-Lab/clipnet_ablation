#!/usr/bin/env python3

"""
Evaluate a CLIPNET training run across loci at each retained training epoch.

The script discovers epoch-numbered checkpoints under a model run root, averages
predictions across all folds for each epoch, and writes performance metrics against
observed PRO-cap signal.
"""

import argparse
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf
from clipnet import utils


EPOCH_RE = re.compile(r"_epoch_(\d+)\.h(?:5|df5)$")
FOLD_DIR_RE = re.compile(r"^f(\d+)$")
FOLD_FILE_RE = re.compile(r"fold[_-]?(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_root",
        type=Path,
        help="Root directory for a model training run, usually containing f1 ... f9.",
    )
    parser.add_argument(
        "fasta",
        type=Path,
        help="FASTA file of 1000 bp sequences, or a preprocessed .npy/.npz array.",
    )
    parser.add_argument(
        "procap",
        type=Path,
        help="CSV(.gz), NPY, or NPZ file of observed PRO-cap signal.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output CSV path for epoch-level performance metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Optional comma-separated epoch list to evaluate, e.g. 5,10,15.",
    )
    parser.add_argument(
        "--prediction_dir",
        type=Path,
        default=None,
        help="Optional directory for averaged prediction .npz files per epoch.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Prediction batch size.",
    )
    parser.add_argument(
        "--allow_missing_folds",
        action="store_true",
        help="Evaluate epochs even if some fold checkpoints are missing.",
    )
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        help="Reverse complement input sequences before prediction.",
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU prediction.")
    parser.add_argument(
        "--use_specific_gpu",
        type=int,
        default=0,
        help="GPU index to use when --gpu is set.",
    )
    return parser.parse_args()


def configure_devices(use_gpu, use_specific_gpu):
    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")
        return
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise ValueError("--gpu was set, but TensorFlow cannot see any GPUs.")
    if use_specific_gpu >= len(gpus):
        raise IndexError(f"Requested GPU index {use_specific_gpu}, found {len(gpus)}.")
    tf.config.set_visible_devices(gpus[use_specific_gpu], "GPU")
    tf.config.experimental.set_memory_growth(gpus[use_specific_gpu], True)


def checkpoint_epoch(path):
    match = EPOCH_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def checkpoint_fold(path, model_root):
    try:
        rel_parts = path.relative_to(model_root).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts[:-1]:
        match = FOLD_DIR_RE.match(part)
        if match is not None:
            return int(match.group(1))
    match = FOLD_FILE_RE.search(path.name)
    if match is not None:
        return int(match.group(1))
    if len(rel_parts) > 1:
        return rel_parts[-2]
    return "root"


def discover_checkpoints(model_root):
    model_root = model_root.resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {model_root}")

    checkpoints = defaultdict(dict)
    fold_ids = set()
    model_files = sorted(
        list(model_root.rglob("*.h5")) + list(model_root.rglob("*.hdf5"))
    )
    for model_fp in model_files:
        epoch = checkpoint_epoch(model_fp)
        if epoch is None:
            continue
        fold = checkpoint_fold(model_fp, model_root)
        if fold in checkpoints[epoch]:
            raise ValueError(
                "Multiple checkpoints found for "
                f"epoch {epoch}, fold {fold}: {checkpoints[epoch][fold]} and {model_fp}"
            )
        checkpoints[epoch][fold] = model_fp
        fold_ids.add(fold)

    if not checkpoints:
        raise FileNotFoundError(
            f"No epoch checkpoints matching *_epoch_###.h5/.hdf5 found under {model_root}"
        )
    return checkpoints, sorted(fold_ids, key=str)


def select_epochs(checkpoints, fold_ids, requested_epochs, allow_missing_folds):
    if requested_epochs is None:
        epochs = sorted(checkpoints)
        if not allow_missing_folds:
            complete_epochs = [
                epoch
                for epoch in epochs
                if all(fold in checkpoints[epoch] for fold in fold_ids)
            ]
            skipped = sorted(set(epochs) - set(complete_epochs))
            if skipped:
                print(
                    "Skipping epochs missing one or more selected folds: "
                    f"{skipped}. Use --allow_missing_folds to evaluate them."
                )
            epochs = complete_epochs
    else:
        epochs = [int(epoch) for epoch in requested_epochs.split(",") if epoch.strip()]
    if not epochs:
        raise ValueError(
            "No epochs selected for evaluation. Use --allow_missing_folds "
            "to evaluate epochs with partial fold coverage."
        )

    selected = []
    for epoch in epochs:
        if epoch not in checkpoints:
            raise FileNotFoundError(f"No checkpoints found for epoch {epoch}.")
        missing = sorted(set(fold_ids) - set(checkpoints[epoch]), key=str)
        if missing and not allow_missing_folds:
            raise FileNotFoundError(
                f"Epoch {epoch} is missing checkpoints for folds: {missing}. "
                "Use --allow_missing_folds to evaluate available folds only."
            )
        selected.append(epoch)
    return selected


def load_sequences(fasta, reverse_complement=False):
    if fasta.suffix == ".npz":
        sequence = np.load(fasta)["arr_0"]
    elif fasta.suffix == ".npy":
        sequence = np.load(fasta)
    else:
        sequence = utils.get_twohot_fasta_sequences(str(fasta))
    if reverse_complement:
        sequence = utils.rc_twohot_het(sequence)
    return sequence


def load_observed(procap):
    if procap.suffix == ".npz":
        return np.load(procap)["arr_0"]
    if procap.suffix == ".npy":
        return np.load(procap)
    if procap.name.endswith(".csv") or procap.name.endswith(".csv.gz"):
        return pd.read_csv(procap, header=None, index_col=0).to_numpy()
    raise ValueError(f"Unsupported PRO-cap file format: {procap}")


def clip_observed_to_prediction(observed, prediction_width):
    if observed.shape[1] < prediction_width:
        raise ValueError(
            f"Observed tracks ({observed.shape[1]}) are shorter than predictions "
            f"({prediction_width})."
        )
    if (observed.shape[1] - prediction_width) % 4 != 0:
        raise ValueError(
            "Padding around observed tracks must be divisible by 4. "
            f"Observed width: {observed.shape[1]}, predicted width: {prediction_width}."
        )

    start = (observed.shape[1] - prediction_width) // 4
    end = observed.shape[1] // 2 - start
    return observed[
        :,
        np.r_[start:end, observed.shape[1] // 2 + start : observed.shape[1] // 2 + end],
    ]


def predict_model(model_fp, sequence, batch_size):
    model = tf.keras.models.load_model(model_fp, compile=False)
    prediction = model.predict(sequence, batch_size=batch_size, verbose=0)
    tf.keras.backend.clear_session()
    profile, quantity = prediction
    if quantity.ndim == 2 and quantity.shape[1] == 1:
        quantity = quantity[:, 0]
    return profile, quantity


def average_epoch_predictions(model_fps, sequence, batch_size):
    profile_sum = None
    quantity_sum = None
    for model_fp in model_fps:
        profile, quantity = predict_model(model_fp, sequence, batch_size)
        if profile_sum is None:
            profile_sum = np.zeros_like(profile, dtype=np.float64)
            quantity_sum = np.zeros_like(quantity, dtype=np.float64)
        profile_sum += profile
        quantity_sum += quantity
    n_models = len(model_fps)
    return profile_sum / n_models, quantity_sum / n_models


def scale_profile(profile, quantity):
    profile_sum = profile.sum(axis=1, keepdims=True)
    normalized_profile = np.divide(
        profile,
        profile_sum,
        out=np.zeros_like(profile, dtype=np.float64),
        where=profile_sum != 0,
    )
    return normalized_profile * quantity.reshape(-1, 1)


def safe_pearson(x, y):
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


def calculate_metrics(profile, quantity, observed):
    if profile.shape[0] != observed.shape[0]:
        raise ValueError(
            f"n predictions ({profile.shape[0]}) and n observed "
            f"({observed.shape[0]}) do not match."
        )
    observed_clipped = clip_observed_to_prediction(observed, profile.shape[1])
    scaled_profile = scale_profile(profile, quantity)

    strand_break = scaled_profile.shape[1] // 2
    pred_directionality = np.log1p(scaled_profile[:, :strand_break].sum(axis=1))
    pred_directionality -= np.log1p(scaled_profile[:, strand_break:].sum(axis=1))
    obs_directionality = np.log1p(observed_clipped[:, :strand_break].sum(axis=1))
    obs_directionality -= np.log1p(observed_clipped[:, strand_break:].sum(axis=1))

    pred_tss = np.concatenate(
        [
            scaled_profile[:, :strand_break].argmax(axis=1),
            scaled_profile[:, strand_break:].argmax(axis=1),
        ]
    )
    obs_tss = np.concatenate(
        [
            observed_clipped[:, :strand_break].argmax(axis=1),
            observed_clipped[:, strand_break:].argmax(axis=1),
        ]
    )

    profile_pearson = pd.DataFrame(scaled_profile).corrwith(
        pd.DataFrame(observed_clipped), axis=1
    )
    profile_js_distance = jensenshannon(scaled_profile, observed_clipped, axis=1)
    observed_quantity = observed_clipped.sum(axis=1)

    return {
        "profile_pearson_median": profile_pearson.median(),
        "profile_pearson_mean": profile_pearson.mean(),
        "profile_pearson_std": profile_pearson.std(),
        "profile_js_distance_median": pd.Series(profile_js_distance).median(),
        "profile_js_distance_mean": pd.Series(profile_js_distance).mean(),
        "profile_js_distance_std": pd.Series(profile_js_distance).std(),
        "directionality_pearson": safe_pearson(
            pred_directionality, obs_directionality
        ),
        "tss_pos_pearson": safe_pearson(pred_tss, obs_tss),
        "quantity_log_pearson": safe_pearson(
            np.log1p(quantity), np.log1p(observed_quantity)
        ),
        "quantity_spearman": spearmanr(quantity, observed_quantity)[0],
    }


def main():
    args = parse_args()
    configure_devices(args.gpu, args.use_specific_gpu)
    checkpoints, fold_ids = discover_checkpoints(args.model_root)
    epochs = select_epochs(
        checkpoints, fold_ids, args.epochs, args.allow_missing_folds
    )

    sequence = load_sequences(args.fasta, reverse_complement=args.reverse_complement)
    observed = load_observed(args.procap)
    if sequence.shape[0] != observed.shape[0]:
        raise ValueError(
            f"n sequences ({sequence.shape[0]}) and n observed "
            f"({observed.shape[0]}) do not match."
        )

    if args.prediction_dir is not None:
        args.prediction_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for epoch in epochs:
        fold_paths = [
            checkpoints[epoch][fold]
            for fold in sorted(checkpoints[epoch], key=str)
        ]
        print(f"Evaluating epoch {epoch} with {len(fold_paths)} fold checkpoints.")
        profile, quantity = average_epoch_predictions(
            fold_paths, sequence, args.batch_size
        )
        metrics = calculate_metrics(profile, quantity, observed)
        rows.append({"epoch": epoch, "n_models": len(fold_paths), **metrics})

        if args.prediction_dir is not None:
            prediction_fp = args.prediction_dir / f"epoch_{epoch:03d}_predictions.npz"
            np.savez_compressed(
                prediction_fp,
                profile=profile,
                quantity=quantity,
                scaled_profile=scale_profile(profile, quantity),
            )

    pd.DataFrame(rows).sort_values("epoch").to_csv(args.output, index=False)
    print(f"Saved evaluation metrics to {args.output}")


if __name__ == "__main__":
    main()
