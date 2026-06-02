#!/usr/bin/env python3

"""
Benchmark training epoch checkpoints on QTL prediction.

Given one model training run root, this script discovers fold-specific epoch
checkpoints, predicts either fold-averaged epoch ensembles or individual fold
checkpoints, splits predictions by allele, and scores predicted QTL effect sizes
against experimental QTL effects.

time python benchmark_epoch_checkpoints.py all \
    ../models/mean_model/ \
    --qtl tiqtl \
    --predictions_root ../predictions/mean_model/tiqtl/ \
    --data_root ../data/ --qtl_data_dir ../data/tiqtl/ \
    --gpu
"""

import argparse
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

QTL_CONFIG = {
    "diqtl": {
        "alleles": "diQTL_snps_per_individual.csv.gz",
        "table": "Table.10a.diQTL.2kb.csv.gz",
        "table_sep": ",",
        "gene_sep": ".",
        "expt": "expt_tracks_per_snp_by_allele.joblib.gz",
    },
    "tiqtl": {
        "alleles": "tiQTL_snps_per_individual.csv.gz",
        "table": "Table.7c.tiQTL.2k.txt.gz",
        "table_sep": "\t",
        "gene_sep": ":",
        "expt": "expt_tracks_per_snp_by_allele.joblib.gz",
    },
}

EPOCH_RE = re.compile(r"_epoch_(\d+)\.h(?:5|df5)$")
FOLD_DIR_RE = re.compile(r"^(?:f|fold_?)(\d+)$")
FOLD_FILE_RE = re.compile(r"(?:^|_)fold_?(\d+)(?:_|\.|$)")
PROFILE_PSEUDOCOUNT = 1e-3


def parse_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["predict", "split", "score", "all"],
        help="Workflow stage to run.",
    )
    parser.add_argument(
        "model_root",
        type=Path,
        help="Model training run root containing fold checkpoint directories.",
    )
    parser.add_argument(
        "--qtl",
        choices=sorted(QTL_CONFIG),
        required=True,
        help="QTL dataset to benchmark.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for output files. Defaults to model_root directory name.",
    )
    parser.add_argument(
        "--mode",
        choices=["ensemble", "folds"],
        default="ensemble",
        help="Inference mode. 'ensemble' averages folds per epoch; 'folds' scores folds separately.",
    )
    parser.add_argument(
        "--epochs",
        type=parse_ints,
        default=None,
        help="Comma-separated epochs to benchmark. Defaults to all discovered epochs.",
    )
    parser.add_argument(
        "--folds",
        type=parse_ints,
        default=None,
        help="Comma-separated fold IDs to use. Defaults to all discovered folds.",
    )
    parser.add_argument(
        "--allow_missing_folds",
        action="store_true",
        help="Benchmark epochs even if some selected folds are missing.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Root containing {qtl}/sequence/{prefix}.fna.gz.",
    )
    parser.add_argument(
        "--predictions_root",
        type=Path,
        default=REPO_ROOT / "predictions",
        help="Root for benchmark prediction and score outputs.",
    )
    parser.add_argument(
        "--qtl_data_dir",
        type=Path,
        default=None,
        help="Directory containing allele matrix, experimental tracks, and QTL table.",
    )
    parser.add_argument(
        "--expt_by_allele",
        type=Path,
        default=None,
        help="Experimental tracks-per-SNP-by-allele joblib file.",
    )
    parser.add_argument(
        "--qtl_table",
        type=Path,
        default=None,
        help="QTL coordinate table with snps and gene columns.",
    )
    parser.add_argument(
        "--fold_assignments",
        type=Path,
        default=REPO_ROOT / "clipnet_data_folds/data_fold_assignments.csv",
        help="Fold assignment CSV used to hold out SNPs in --mode folds.",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default=None,
        help="Optional comma-separated PRO-cap prefixes. Defaults to allele matrix rows.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Prediction batch size.",
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU prediction.")
    parser.add_argument(
        "--use_specific_gpu",
        type=int,
        default=0,
        help="GPU index to use when --gpu is set.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print prediction targets without loading models or writing files.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        help="HDF5 compression for prediction outputs. Default: gzip.",
    )
    return parser.parse_args()


def qtl_data_dir(args):
    return args.qtl_data_dir or SCRIPT_DIR / args.qtl


def benchmark_root(args):
    run_name = args.run_name or args.model_root.resolve().name
    return args.predictions_root / args.qtl / "epoch_checkpoint_benchmark" / run_name


def output_root(args):
    if args.mode == "ensemble":
        return benchmark_root(args)
    return benchmark_root(args) / args.mode


def allele_matrix_path(args):
    return qtl_data_dir(args) / QTL_CONFIG[args.qtl]["alleles"]


def load_allele_matrix(args):
    return pd.read_csv(allele_matrix_path(args), index_col=0)


def selected_prefixes(args, allele_matrix):
    if args.prefixes is None:
        return list(allele_matrix.index)
    return [prefix for prefix in args.prefixes.split(",") if prefix.strip()]


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


def discover_checkpoints(args):
    model_root = args.model_root.resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {model_root}")

    checkpoints = defaultdict(dict)
    for model_fp in sorted(
        list(model_root.rglob("*.h5")) + list(model_root.rglob("*.hdf5"))
    ):
        epoch = checkpoint_epoch(model_fp)
        if epoch is None:
            continue
        fold = checkpoint_fold(model_fp, model_root)
        if args.folds is not None and fold not in args.folds:
            continue
        if fold in checkpoints[epoch]:
            raise ValueError(
                "Multiple checkpoints found for "
                f"epoch {epoch}, fold {fold}: {checkpoints[epoch][fold]} and {model_fp}"
            )
        checkpoints[epoch][fold] = model_fp

    if not checkpoints:
        raise FileNotFoundError(
            f"No epoch checkpoints matching *_epoch_###.h5/.hdf5 found under {model_root}"
        )
    return checkpoints


def selected_epochs(args, checkpoints):
    epochs = sorted(checkpoints) if args.epochs is None else args.epochs
    missing = [epoch for epoch in epochs if epoch not in checkpoints]
    if missing:
        raise FileNotFoundError(f"No checkpoints found for epochs: {missing}")
    return epochs


def selected_folds(args, checkpoints):
    if args.folds is not None:
        return args.folds
    folds = set()
    for epoch_checkpoints in checkpoints.values():
        folds.update(epoch_checkpoints)
    return sorted(folds, key=str)


def epoch_checkpoint_paths(args, checkpoints, epoch, folds):
    missing = [fold for fold in folds if fold not in checkpoints[epoch]]
    if missing and not args.allow_missing_folds:
        raise FileNotFoundError(
            f"Epoch {epoch} is missing checkpoints for folds: {missing}. "
            "Use --allow_missing_folds to average available folds only."
        )
    return [checkpoints[epoch][fold] for fold in folds if fold in checkpoints[epoch]]


def prediction_path(args, epoch, prefix):
    root = output_root(args) / "predictions" / f"epoch_{epoch:03d}"
    return root / f"{prefix}.h5"


def fold_prediction_path(args, epoch, fold, prefix):
    root = (
        output_root(args)
        / "predictions"
        / f"epoch_{epoch:03d}"
        / f"fold_{fold}"
    )
    return root / f"{prefix}.h5"


def split_path(args, epoch):
    return (
        output_root(args)
        / "split_by_allele"
        / f"epoch_{epoch:03d}_pred_per_snp_by_allele.joblib.gz"
    )


def fold_split_path(args, epoch, fold):
    return (
        output_root(args)
        / "split_by_allele"
        / f"epoch_{epoch:03d}_fold_{fold}_pred_per_snp_by_allele.joblib.gz"
    )


def score_path(args, epoch):
    return (
        output_root(args)
        / "scores"
        / f"epoch_{epoch:03d}_l2_scores.csv.gz"
    )


def fold_score_path(args, epoch, fold):
    return (
        output_root(args)
        / "scores"
        / f"epoch_{epoch:03d}_fold_{fold}_l2_scores.csv.gz"
    )


def summary_path(args):
    return output_root(args) / "epoch_qtl_benchmark_summary.csv"


def configure_prediction_runtime(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    import tensorflow as tf

    if not args.gpu:
        tf.config.set_visible_devices([], "GPU")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if args.use_specific_gpu >= len(gpus):
            raise IndexError(
                f"Requested GPU {args.use_specific_gpu}, but TensorFlow sees {len(gpus)}."
            )
        tf.config.set_visible_devices(gpus[args.use_specific_gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[args.use_specific_gpu], True)


def create_predictor(args):
    if args.dry_run:
        return None
    configure_prediction_runtime(args)
    from clipnet import clipnet

    return clipnet.CLIPNET(
        n_gpus=1 if args.gpu else 0,
        use_specific_gpu=args.use_specific_gpu if args.gpu else -1,
    )


def average_predictions(nn, checkpoint_paths, sequence_fp, args):
    profile_sum = None
    quantity_sum = None

    for checkpoint_fp in checkpoint_paths:
        prediction = nn.predict_on_fasta(
            model_fp=str(checkpoint_fp),
            fasta_fp=str(sequence_fp),
            low_mem=True,
            silence=True,
        )
        profile, quantity = prediction
        if quantity.ndim == 2 and quantity.shape[1] == 1:
            quantity = quantity[:, 0]
        scaled_profile = scale_profile(profile, quantity)
        if profile_sum is None:
            profile_sum = np.zeros_like(scaled_profile, dtype=np.float64)
            quantity_sum = np.zeros_like(quantity, dtype=np.float64)
        profile_sum += scaled_profile
        quantity_sum += quantity

    n_models = len(checkpoint_paths)
    return profile_sum / n_models, quantity_sum / n_models


def predict_checkpoint(nn, checkpoint_fp, sequence_fp):
    prediction = nn.predict_on_fasta(
        model_fp=str(checkpoint_fp),
        fasta_fp=str(sequence_fp),
        low_mem=True,
        silence=True,
    )
    profile, quantity = prediction
    if quantity.ndim == 2 and quantity.shape[1] == 1:
        quantity = quantity[:, 0]
    return profile, quantity


def scale_profile(track, quantity):
    return (
        track / (track.sum(axis=1, keepdims=True) + PROFILE_PSEUDOCOUNT)
    ) * quantity[:, None]


def run_predict(args):
    allele_matrix = load_allele_matrix(args)
    prefixes = selected_prefixes(args, allele_matrix)
    checkpoints = discover_checkpoints(args)
    epochs = selected_epochs(args, checkpoints)
    folds = selected_folds(args, checkpoints)
    sequence_root = args.data_root / args.qtl / "sequence"
    nn = create_predictor(args)

    for epoch in epochs:
        checkpoint_paths = epoch_checkpoint_paths(args, checkpoints, epoch, folds)
        if args.mode == "ensemble":
            print(f"Epoch {epoch}: averaging {len(checkpoint_paths)} fold checkpoints.")
        else:
            print(f"Epoch {epoch}: predicting {len(checkpoint_paths)} fold checkpoints.")

        for prefix in tqdm.tqdm(prefixes, desc=f"Predicting epoch {epoch}"):
            sequence_fp = sequence_root / f"{prefix}.fna.gz"
            if args.mode == "ensemble":
                output_fp = prediction_path(args, epoch, prefix)
                if args.dry_run:
                    print(f"predict {checkpoint_paths} {sequence_fp} -> {output_fp}")
                    continue
                if args.skip_existing and output_fp.exists():
                    continue
                output_fp.parent.mkdir(parents=True, exist_ok=True)
                profile, quantity = average_predictions(
                    nn, checkpoint_paths, sequence_fp, args
                )
                import h5py

                with h5py.File(output_fp, "w") as hf:
                    hf.create_dataset("track", data=profile, compression=args.compression)
                    hf.create_dataset(
                        "quantity", data=quantity, compression=args.compression
                    )
                    hf.attrs["track_is_scaled"] = True
            else:
                for fold in folds:
                    if fold not in checkpoints[epoch]:
                        continue
                    output_fp = fold_prediction_path(args, epoch, fold, prefix)
                    checkpoint_fp = checkpoints[epoch][fold]
                    if args.dry_run:
                        print(f"predict {checkpoint_fp} {sequence_fp} -> {output_fp}")
                        continue
                    if args.skip_existing and output_fp.exists():
                        continue
                    output_fp.parent.mkdir(parents=True, exist_ok=True)
                    profile, quantity = predict_checkpoint(nn, checkpoint_fp, sequence_fp)
                    import h5py

                    with h5py.File(output_fp, "w") as hf:
                        hf.create_dataset(
                            "track", data=profile, compression=args.compression
                        )
                        hf.create_dataset(
                            "quantity", data=quantity, compression=args.compression
                        )
                        hf.attrs["track_is_scaled"] = False


def scaled_prediction(pred_fp):
    import h5py

    with h5py.File(pred_fp, "r") as pred:
        track = pred["track"][:]
        quantity = pred["quantity"][:]
        track_is_scaled = bool(pred.attrs.get("track_is_scaled", False))
    if quantity.ndim == 2 and quantity.shape[1] == 1:
        quantity = quantity[:, 0]
    scaled = track if track_is_scaled else scale_profile(track, quantity)
    return scaled, quantity


def split_epoch_predictions(args, epoch, allele_matrix, prefixes, snps, fold=None):
    out_fp = split_path(args, epoch) if fold is None else fold_split_path(args, epoch, fold)
    if args.skip_existing and out_fp.exists():
        return

    pred_tracks = {snp: [[], [], []] for snp in snps}
    pred_quantities = {snp: [[], [], []] for snp in snps}

    for prefix in prefixes:
        pred_fp = (
            prediction_path(args, epoch, prefix)
            if fold is None
            else fold_prediction_path(args, epoch, fold, prefix)
        )
        track, quantity = scaled_prediction(pred_fp)
        for snp_idx, snp in enumerate(snps):
            allele = allele_matrix.at[prefix, snp]
            if allele == 0:
                allele_idx = 0
            elif allele == 1:
                allele_idx = 1
            elif allele == 0.5:
                allele_idx = 2
            else:
                continue
            pred_tracks[snp][allele_idx].append(track[snp_idx])
            pred_quantities[snp][allele_idx].append(quantity[snp_idx])

    pred_tracks = {
        snp: [np.array(values) for values in allele_values]
        for snp, allele_values in pred_tracks.items()
    }
    pred_quantities = {
        snp: [np.array(values) for values in allele_values]
        for snp, allele_values in pred_quantities.items()
    }

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump([pred_tracks, pred_quantities], out_fp)


def run_split(args):
    allele_matrix = load_allele_matrix(args)
    prefixes = selected_prefixes(args, allele_matrix)
    snps = list(allele_matrix.columns)
    checkpoints = discover_checkpoints(args)
    epochs = selected_epochs(args, checkpoints)
    folds = selected_folds(args, checkpoints)

    for epoch in tqdm.tqdm(epochs, desc="Splitting epoch predictions by allele"):
        if args.mode == "ensemble":
            split_epoch_predictions(args, epoch, allele_matrix, prefixes, snps)
        else:
            epoch_checkpoint_paths(args, checkpoints, epoch, folds)
            for fold in folds:
                if fold in checkpoints[epoch]:
                    split_epoch_predictions(
                        args, epoch, allele_matrix, prefixes, snps, fold=fold
                    )


def l2_score(x, y):
    return np.sqrt(np.sum(np.square(x - y), axis=1).astype(np.float32))


def finite_correlation(x, y, method):
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    return pd.Series(x[finite]).corr(pd.Series(y[finite]), method=method)


def strict_correlation(x, y, method):
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan
    if len(x) < 2:
        return np.nan
    return pd.Series(x).corr(pd.Series(y), method=method)


def load_experimental_tracks(args):
    import joblib

    config = QTL_CONFIG[args.qtl]
    expt_fp = args.expt_by_allele or qtl_data_dir(args) / config["expt"]
    expt = joblib.load(expt_fp)
    expt = {
        snp: values
        for snp, values in expt.items()
        if values[0] is not None and values[1] is not None
    }
    snps = list(expt)
    ref = pd.DataFrame({snp: expt[snp][0].mean(axis=0) for snp in snps}).transpose()
    alt = pd.DataFrame({snp: expt[snp][1].mean(axis=0) for snp in snps}).transpose()
    return expt, ref, alt


def load_qtl_coordinates(args, snps):
    config = QTL_CONFIG[args.qtl]
    table_fp = args.qtl_table or qtl_data_dir(args) / config["table"]
    qtls = pd.DataFrame({"snps": snps})
    qtl_table = pd.read_csv(table_fp, sep=config["table_sep"]).drop_duplicates(
        subset="snps", keep="first"
    )
    qtl_coord = pd.merge(
        qtls,
        qtl_table,
        on="snps",
        how="left",
    )
    qtl_coord[["chrom", "start"]] = qtl_coord["gene"].str.split(
        config["gene_sep"], expand=True
    )
    return qtl_coord


def holdout_snps(args, fold, qtl_coord):
    data_splits = pd.read_csv(args.fold_assignments)
    holdout_chroms = set(data_splits[data_splits.fold == int(fold)].chrom)
    return set(qtl_coord[qtl_coord.chrom.isin(holdout_chroms)]["snps"])


def score_epoch(args, epoch, snps, expt_ref, expt_alt, qtl_coord, fold=None):
    out_fp = score_path(args, epoch) if fold is None else fold_score_path(args, epoch, fold)
    if args.skip_existing and out_fp.exists():
        scores = pd.read_csv(out_fp, index_col=0)
    else:
        import joblib

        pred_fp = split_path(args, epoch) if fold is None else fold_split_path(args, epoch, fold)
        pred = joblib.load(pred_fp)[0]
        pred_ref = pd.DataFrame(
            {snp: np.mean(pred[snp][0], axis=0) for snp in snps}
        ).transpose()
        pred_alt = pd.DataFrame(
            {snp: np.mean(pred[snp][1], axis=0) for snp in snps}
        ).transpose()
        scores = pd.DataFrame(
            {
                "expt": l2_score(expt_ref.to_numpy(), expt_alt.to_numpy()),
                "pred": l2_score(pred_ref.to_numpy(), pred_alt.to_numpy()),
            },
            index=expt_ref.index,
        )
        qtl_index = qtl_coord.set_index("snps")
        scores["chrom"] = qtl_index.loc[scores.index, "chrom"]
        scores["start"] = qtl_index.loc[scores.index, "start"]
        if fold is not None:
            keep = holdout_snps(args, fold, qtl_coord)
            scores = scores.loc[[snp for snp in scores.index if snp in keep]]
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out_fp)

    expt_values = scores["expt"].to_numpy()
    pred_values = scores["pred"].to_numpy()
    finite = np.isfinite(expt_values) & np.isfinite(pred_values)

    return {
        "mode": args.mode,
        "epoch": epoch,
        "fold": np.nan if fold is None else fold,
        "n_snps": scores.shape[0],
        "n_valid_snps": int(finite.sum()),
        "n_invalid_snps": int((~finite).sum()),
        "l2_pearson": strict_correlation(expt_values, pred_values, "pearson"),
        "l2_spearman": strict_correlation(expt_values, pred_values, "spearman"),
        "l2_pearson_valid": finite_correlation(expt_values, pred_values, "pearson"),
        "l2_spearman_valid": finite_correlation(expt_values, pred_values, "spearman"),
        "pred_l2_mean": scores["pred"].mean(),
        "expt_l2_mean": scores["expt"].mean(),
    }


def run_score(args):
    expt, expt_ref, expt_alt = load_experimental_tracks(args)
    snps = list(expt)
    qtl_coord = load_qtl_coordinates(args, snps)
    checkpoints = discover_checkpoints(args)
    epochs = selected_epochs(args, checkpoints)
    folds = selected_folds(args, checkpoints)

    summary_rows = []
    for epoch in tqdm.tqdm(epochs, desc="Scoring epoch QTL predictions"):
        if args.mode == "ensemble":
            summary_rows.append(
                score_epoch(args, epoch, snps, expt_ref, expt_alt, qtl_coord)
            )
        else:
            epoch_checkpoint_paths(args, checkpoints, epoch, folds)
            for fold in folds:
                if fold in checkpoints[epoch]:
                    summary_rows.append(
                        score_epoch(
                            args, epoch, snps, expt_ref, expt_alt, qtl_coord, fold=fold
                        )
                    )

    out_fp = summary_path(args)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).sort_values("epoch").to_csv(out_fp, index=False)
    print(f"Saved epoch benchmark summary to {out_fp}")


def main():
    args = parse_args()
    if args.command in {"predict", "all"}:
        run_predict(args)
    if args.command in {"split", "all"}:
        run_split(args)
    if args.command in {"score", "all"}:
        run_score(args)


if __name__ == "__main__":
    main()
