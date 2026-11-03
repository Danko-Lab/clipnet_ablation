#!/usr/bin/env python3

"""Per-fold test-set performance of CLIPNET and reference models.

Each fold model is scored on its own held-out test fold (concat_*_{k}.npz):
personal-genome sequences for ensemble_models, reference sequences for
reference_models, against the same PRO-cap tracks. Metrics follow
calculate_performance_metrics.py (median track Pearson, directionality, TSS
position, log-quantity Pearson, quantity Spearman), plus the fraction of loci
where the ReLU count head predicts exactly 0.

Defaults to the main models; --clipnet_models / --ref_models take path
templates with {fold} to score other runs, e.g. retrained best checkpoints.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

MAIN_ROOT = "/fs/cbsubscb17/storage/projects/CLIPNET"
SEQUENCES = {
    "clipnet": "concat_sequence_{fold}.npz",
    "ref_model": "concat_sequence_reference_{fold}.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clipnet_models",
                        default=f"{MAIN_ROOT}/ensemble_models/fold_{{fold}}.h5")
    parser.add_argument("--ref_models",
                        default=f"{MAIN_ROOT}/reference_models/fold_{{fold}}.h5")
    parser.add_argument("--data_dir", type=Path, default=Path("../data"))
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8,9")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("main_fold_test_metrics.csv"))
    return parser.parse_args()


def predict(model, sequence, batch_size):
    tracks, quantities = [], []
    for i in range(0, len(sequence), batch_size):
        track, quantity = model.predict(sequence[i : i + batch_size].astype(np.float32), verbose=0)
        tracks.append(track.astype(np.float32))
        quantities.append(quantity.reshape(-1))
    return np.concatenate(tracks), np.concatenate(quantities)


def row_pearson(x, y, chunk=20000):
    out = np.empty(len(x))
    for i in range(0, len(x), chunk):
        a = x[i : i + chunk] - x[i : i + chunk].mean(1, keepdims=True)
        b = y[i : i + chunk] - y[i : i + chunk].mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i : i + chunk] = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return out


def metrics(track, quantity, observed):
    start = (observed.shape[1] - track.shape[1]) // 4
    end = observed.shape[1] // 2 - start
    half = observed.shape[1] // 2
    obs = observed[:, np.r_[start:end, half + start : half + end]].astype(np.float32)
    strand = track.shape[1] // 2
    track_dir = np.log1p(track[:, :strand].sum(1)) - np.log1p(track[:, strand:].sum(1))
    obs_dir = np.log1p(obs[:, :strand].sum(1)) - np.log1p(obs[:, strand:].sum(1))
    pred_tss = np.concatenate([track[:, :strand].argmax(1), track[:, strand:].argmax(1)])
    obs_tss = np.concatenate([obs[:, :strand].argmax(1), obs[:, strand:].argmax(1)])
    obs_sum = obs.sum(1)
    return {
        "n_loci": len(track),
        "track_pearson_median": float(np.nanmedian(row_pearson(track, obs))),
        "directionality_pearson": pearsonr(track_dir, obs_dir)[0],
        "tss_pos_pearson": pearsonr(pred_tss, obs_tss)[0],
        "quantity_log_pearson": pearsonr(np.log1p(quantity), np.log1p(obs_sum))[0],
        "quantity_spearman": spearmanr(quantity, obs_sum)[0],
        "frac_quantity_zero": float((quantity == 0).mean()),
    }


def main():
    args = parse_args()
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    rows = []
    for fold in (int(f) for f in args.folds.split(",")):
        observed = np.load(args.data_dir / f"concat_procap_{fold}.npz")["arr_0"]
        templates = {"clipnet": args.clipnet_models, "ref_model": args.ref_models}
        for arm, seq_name in SEQUENCES.items():
            sequence = np.load(args.data_dir / seq_name.format(fold=fold))["arr_0"].astype(np.int8)
            model = tf.keras.models.load_model(templates[arm].format(fold=fold), compile=False)
            track, quantity = predict(model, sequence, args.batch_size)
            row = {"fold": fold, "arm": arm, **metrics(track, quantity, observed)}
            print(row, flush=True)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            del sequence, model, track, quantity
            tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
