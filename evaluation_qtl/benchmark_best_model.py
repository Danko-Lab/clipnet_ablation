#!/usr/bin/env python3

"""
Benchmark stable best-model checkpoints on QTL prediction.

Given one model training run root, this script discovers fold-specific best
checkpoints, predicts either fold-averaged best-model ensembles or individual
fold checkpoints, splits predictions by allele, and scores predicted QTL effect
sizes against experimental QTL effects.
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

try:
    from qtl_filters import (
        build_filter_report,
        eligible_snps,
        load_prefix_map,
        print_filter_summary,
        write_filter_report,
    )
except ImportError:
    from evaluation_qtl.qtl_filters import (
        build_filter_report,
        eligible_snps,
        load_prefix_map,
        print_filter_summary,
        write_filter_report,
    )


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

BEST_RE = re.compile(r"_(resume_)?best\.h(?:5|df5)$")
FOLD_DIR_RE = re.compile(r"^(?:f|fold_?)(\d+)$")
FOLD_FILE_RE = re.compile(r"(?:^|_)fold_?(\d+)(?:_|\.|$)")
PROFILE_PSEUDOCOUNT = 1e-3
L2_LOG_PSEUDOCOUNT = 1e-3


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
        help="Inference mode. 'ensemble' averages fold best checkpoints; 'folds' scores folds separately.",
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
        help="Benchmark best checkpoints even if some selected folds are missing.",
    )
    parser.add_argument(
        "--best_variant",
        choices=["auto", "fresh", "resume"],
        default="auto",
        help=(
            "Which stable best checkpoint to use. 'fresh' selects *_best.h5/.hdf5, "
            "'resume' selects *_resume_best.h5/.hdf5, and 'auto' errors if both "
            "exist for the same fold."
        ),
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
        "--prefix_map",
        type=Path,
        default=REPO_ROOT / "data_spec/procap_to_1k_genomes.json",
        help="JSON mapping PRO-cap library prefixes to unique individuals.",
    )
    parser.add_argument(
        "--pvalue_column",
        default=None,
        help="QTL-table p-value column. Auto-detected when omitted.",
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
        "--allow_row_order",
        action="store_true",
        help=(
            "Allow prediction rows to be matched to allele-matrix SNP columns by "
            "order when stored sequence IDs do not match SNP IDs. Requires equal "
            "prediction-row and SNP counts."
        ),
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
    return args.predictions_root / args.qtl / "best_model_benchmark" / run_name


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


def fasta_sequence_ids(fasta_fp):
    import gzip

    opener = gzip.open if str(fasta_fp).endswith(".gz") else open
    sequence_ids = []
    with opener(fasta_fp, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                sequence_ids.append(line[1:].strip().split()[0])
    return sequence_ids


def write_prediction_h5(output_fp, profile, quantity, compression, track_is_scaled, sequence_ids):
    import h5py

    if sequence_ids is not None and len(sequence_ids) != profile.shape[0]:
        raise ValueError(
            f"Number of FASTA records ({len(sequence_ids)}) does not match "
            f"prediction rows ({profile.shape[0]}) for {output_fp}."
        )
    with h5py.File(output_fp, "w") as hf:
        hf.create_dataset("track", data=profile, compression=compression)
        hf.create_dataset("quantity", data=quantity, compression=compression)
        hf.attrs["track_is_scaled"] = track_is_scaled
        if sequence_ids is not None:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            hf.create_dataset(
                "sequence_ids",
                data=np.asarray(sequence_ids, dtype=object),
                dtype=string_dtype,
            )


def decode_sequence_ids(values):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def match_sequence_id(sequence_id, snp_to_idx):
    candidates = [sequence_id, sequence_id.split()[0]]
    for delimiter in ("|", ",", ";"):
        candidates.extend(part for part in sequence_id.split(delimiter) if part)
    for candidate in candidates:
        if candidate in snp_to_idx:
            return candidate

    matches = [snp for snp in snp_to_idx if snp in sequence_id]
    if len(matches) == 1:
        return matches[0]
    return None


def prediction_row_order(pred_fp, snps, allow_row_order=False):
    import h5py

    with h5py.File(pred_fp, "r") as pred:
        if "sequence_ids" not in pred:
            return list(enumerate(snps))
        sequence_ids = decode_sequence_ids(pred["sequence_ids"][:])

    snp_to_idx = {snp: idx for idx, snp in enumerate(snps)}
    rows = []
    unmatched = []
    seen = set()
    for row_idx, sequence_id in enumerate(sequence_ids):
        snp = match_sequence_id(sequence_id, snp_to_idx)
        if snp is None:
            unmatched.append(sequence_id)
            continue
        if snp in seen:
            raise ValueError(
                f"Duplicate sequence records map to SNP {snp} in {pred_fp}."
            )
        seen.add(snp)
        rows.append((row_idx, snp))

    missing = [snp for snp in snps if snp not in seen]
    if unmatched or missing:
        if allow_row_order and not rows and len(sequence_ids) == len(snps):
            logging.warning(
                "Prediction sequence IDs in %s do not match allele-matrix SNP IDs; "
                "using row-order mapping because --allow_row_order was set.",
                pred_fp,
            )
            return list(enumerate(snps))
        raise ValueError(
            f"Prediction sequence IDs in {pred_fp} do not match the allele matrix. "
            f"Unmatched sequence IDs: {unmatched[:5]} "
            f"({len(unmatched)} total). Missing SNPs: {missing[:5]} "
            f"({len(missing)} total). If these are the original CLIPNET QTL "
            f"windows in allele-matrix order, rerun with --allow_row_order."
        )
    return rows


def checkpoint_variant(path):
    match = BEST_RE.search(path.name)
    if match is None:
        return None
    return "resume" if match.group(1) else "fresh"


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

    checkpoints_by_fold = defaultdict(dict)
    for model_fp in sorted(
        list(model_root.rglob("*.h5")) + list(model_root.rglob("*.hdf5"))
    ):
        variant = checkpoint_variant(model_fp)
        if variant is None:
            continue
        fold = checkpoint_fold(model_fp, model_root)
        if args.folds is not None and fold not in args.folds:
            continue
        if variant in checkpoints_by_fold[fold]:
            raise ValueError(
                f"Multiple {variant} best checkpoints found for fold {fold}: "
                f"{checkpoints_by_fold[fold][variant]} and {model_fp}"
            )
        checkpoints_by_fold[fold][variant] = model_fp

    checkpoints = {}
    for fold, variants in checkpoints_by_fold.items():
        if args.best_variant == "auto":
            if len(variants) > 1:
                raise ValueError(
                    f"Both fresh and resume best checkpoints found for fold {fold}: "
                    f"{variants}. Use --best_variant fresh or --best_variant resume."
                )
            checkpoints[fold] = next(iter(variants.values()))
        elif args.best_variant in variants:
            checkpoints[fold] = variants[args.best_variant]

    if not checkpoints:
        raise FileNotFoundError(
            f"No {args.best_variant} best checkpoints matching *_best.h5/.hdf5 "
            f"or *_resume_best.h5/.hdf5 found under {model_root}"
        )
    return checkpoints


def selected_folds(args, checkpoints):
    if args.folds is not None:
        return args.folds
    return sorted(checkpoints, key=str)


def checkpoint_paths(args, checkpoints, folds):
    missing = [fold for fold in folds if fold not in checkpoints]
    if missing and not args.allow_missing_folds:
        raise FileNotFoundError(
            f"Best-model checkpoints are missing for folds: {missing}. "
            "Use --allow_missing_folds to average available folds only."
        )
    return [checkpoints[fold] for fold in folds if fold in checkpoints]


def prediction_path(args, prefix):
    root = output_root(args) / "predictions" / "best_model"
    return root / f"{prefix}.h5"


def fold_prediction_path(args, fold, prefix):
    root = (
        output_root(args)
        / "predictions"
        / "best_model"
        / f"fold_{fold}"
    )
    return root / f"{prefix}.h5"


def split_path(args):
    return (
        output_root(args)
        / "split_by_allele"
        / "best_model_pred_per_snp_by_allele.joblib.gz"
    )


def ensemble_split_path(args):
    return (
        benchmark_root(args)
        / "split_by_allele"
        / "best_model_pred_per_snp_by_allele.joblib.gz"
    )


def fold_split_path(args, fold):
    return (
        output_root(args)
        / "split_by_allele"
        / f"best_model_fold_{fold}_pred_per_snp_by_allele.joblib.gz"
    )


def score_path(args):
    return (
        output_root(args)
        / "scores"
        / "best_model_l2_scores.csv.gz"
    )


def ensemble_score_path(args):
    return (
        benchmark_root(args)
        / "scores"
        / "best_model_l2_scores.csv.gz"
    )


def fold_score_path(args, fold):
    return (
        output_root(args)
        / "scores"
        / f"best_model_fold_{fold}_l2_scores.csv.gz"
    )


def summary_path(args):
    return output_root(args) / "best_model_qtl_benchmark_summary.csv"


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
        if profile_sum is None:
            profile_sum = np.zeros_like(profile, dtype=np.float64)
            quantity_sum = np.zeros_like(quantity, dtype=np.float64)
        profile_sum += profile
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
    folds = selected_folds(args, checkpoints)
    sequence_root = args.data_root / args.qtl / "sequence"
    nn = create_predictor(args)
    selected_checkpoint_paths = checkpoint_paths(args, checkpoints, folds)

    if args.mode == "ensemble":
        print(f"Best model: averaging {len(selected_checkpoint_paths)} fold checkpoints.")
    else:
        print(f"Best model: predicting {len(selected_checkpoint_paths)} fold checkpoints.")

    for prefix in tqdm.tqdm(prefixes, desc="Predicting best model"):
        sequence_fp = sequence_root / f"{prefix}.fna.gz"
        sequence_ids = None if args.dry_run else fasta_sequence_ids(sequence_fp)
        if args.mode == "ensemble":
            output_fp = prediction_path(args, prefix)
            if args.dry_run:
                print(f"predict {selected_checkpoint_paths} {sequence_fp} -> {output_fp}")
                continue
            if args.skip_existing and output_fp.exists():
                continue
            output_fp.parent.mkdir(parents=True, exist_ok=True)
            profile, quantity = average_predictions(
                nn, selected_checkpoint_paths, sequence_fp, args
            )
            write_prediction_h5(
                output_fp,
                profile,
                quantity,
                args.compression,
                track_is_scaled=False,
                sequence_ids=sequence_ids,
            )
        else:
            for fold in folds:
                if fold not in checkpoints:
                    continue
                output_fp = fold_prediction_path(args, fold, prefix)
                checkpoint_fp = checkpoints[fold]
                if args.dry_run:
                    print(f"predict {checkpoint_fp} {sequence_fp} -> {output_fp}")
                    continue
                if args.skip_existing and output_fp.exists():
                    continue
                output_fp.parent.mkdir(parents=True, exist_ok=True)
                profile, quantity = predict_checkpoint(nn, checkpoint_fp, sequence_fp)
                write_prediction_h5(
                    output_fp,
                    profile,
                    quantity,
                    args.compression,
                    track_is_scaled=False,
                    sequence_ids=sequence_ids,
                )


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


def split_best_predictions(args, allele_matrix, prefixes, snps, fold=None):
    out_fp = split_path(args) if fold is None else fold_split_path(args, fold)
    if args.skip_existing and out_fp.exists():
        return

    pred_tracks = {snp: [[], [], []] for snp in snps}
    pred_quantities = {snp: [[], [], []] for snp in snps}

    for prefix in prefixes:
        pred_fp = (
            prediction_path(args, prefix)
            if fold is None
            else fold_prediction_path(args, fold, prefix)
        )
        track, quantity = scaled_prediction(pred_fp)
        for row_idx, snp in prediction_row_order(
            pred_fp, snps, allow_row_order=args.allow_row_order
        ):
            allele = allele_matrix.at[prefix, snp]
            if allele == 0:
                allele_idx = 0
            elif allele == 1:
                allele_idx = 1
            elif allele == 0.5:
                allele_idx = 2
            else:
                continue
            pred_tracks[snp][allele_idx].append(track[row_idx])
            pred_quantities[snp][allele_idx].append(quantity[row_idx])

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
    folds = selected_folds(args, checkpoints)

    if args.mode == "ensemble":
        split_best_predictions(args, allele_matrix, prefixes, snps)
    else:
        checkpoint_paths(args, checkpoints, folds)
        for fold in tqdm.tqdm(folds, desc="Splitting best-model fold predictions by allele"):
            if fold in checkpoints:
                split_best_predictions(args, allele_matrix, prefixes, snps, fold=fold)


def l2_score(x, y):
    return np.sqrt(np.sum(np.square(x - y), axis=1).astype(np.float32))


def log_l2_values(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0, np.log(values + L2_LOG_PSEUDOCOUNT), np.nan)


def finite_correlation(x, y, method):
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    return correlation(x[finite], y[finite], method)


def strict_correlation(x, y, method):
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan
    if len(x) < 2:
        return np.nan
    return correlation(x, y, method)


def correlation(x, y, method):
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    if method == "spearman":
        x_series = x_series.rank()
        y_series = y_series.rank()
    return x_series.corr(y_series, method="pearson")


def summarize_scores(args, scores, fold, aggregation):
    expt_values = scores["expt"].to_numpy()
    pred_values = scores["pred"].to_numpy()
    expt_log_values = log_l2_values(expt_values)
    pred_log_values = log_l2_values(pred_values)
    finite = np.isfinite(expt_log_values) & np.isfinite(pred_log_values)

    return {
        "mode": args.mode,
        "aggregation": aggregation,
        "checkpoint": getattr(args, "checkpoint_label", "best_model"),
        "fold": fold,
        "n_snps": scores.shape[0],
        "n_valid_snps": int(finite.sum()),
        "n_invalid_snps": int((~finite).sum()),
        "l2_pearson": strict_correlation(expt_values, pred_values, "pearson"),
        "l2_spearman": strict_correlation(expt_values, pred_values, "spearman"),
        "l2_pearson_valid": finite_correlation(expt_values, pred_values, "pearson"),
        "l2_spearman_valid": finite_correlation(expt_values, pred_values, "spearman"),
        "log_l2_pearson": strict_correlation(expt_log_values, pred_log_values, "pearson"),
        "log_l2_pearson_valid": finite_correlation(expt_log_values, pred_log_values, "pearson"),
        "manuscript_pearson": strict_correlation(
            expt_log_values, pred_log_values, "pearson"
        ),
        "pred_l2_mean": scores["pred"].mean(),
        "pred_l2_std": scores["pred"].std(),
        "expt_l2_mean": scores["expt"].mean(),
        "expt_l2_std": scores["expt"].std(),
        "pred_log_l2_mean": np.nanmean(pred_log_values),
        "pred_log_l2_std": np.nanstd(pred_log_values, ddof=1),
        "expt_log_l2_mean": np.nanmean(expt_log_values),
        "expt_log_l2_std": np.nanstd(expt_log_values, ddof=1),
    }


def load_experimental_tracks(args):
    import joblib

    config = QTL_CONFIG[args.qtl]
    expt_fp = args.expt_by_allele or qtl_data_dir(args) / config["expt"]
    expt = joblib.load(expt_fp)
    if isinstance(expt, (list, tuple)):
        if len(expt) != 2 or not isinstance(expt[0], dict):
            raise TypeError(f"Unsupported experimental track format in {expt_fp}.")
        logging.warning(
            "Using tracks from prediction-style [tracks, quantities] file %s. "
            "Regenerate it with generate_expt_tracks_by_allele.py when convenient.",
            expt_fp,
        )
        expt = expt[0]
    if not isinstance(expt, dict):
        raise TypeError(f"Expected a SNP-to-allele dictionary in {expt_fp}.")
    available = {
        snp
        for snp, values in expt.items()
        if values[0] is not None
        and values[1] is not None
        and np.asarray(values[0]).size
        and np.asarray(values[1]).size
    }
    config = QTL_CONFIG[args.qtl]
    table_fp = args.qtl_table or qtl_data_dir(args) / config["table"]
    qtl_table = pd.read_csv(table_fp, sep=config["table_sep"])
    allele_matrix = load_allele_matrix(args)
    report = build_filter_report(
        args.qtl,
        allele_matrix,
        qtl_table,
        load_prefix_map(args.prefix_map),
        pvalue_column=args.pvalue_column,
        available_experimental_snps=available,
    )
    report_fp = benchmark_root(args) / "scores" / "qtl_filter_report.csv.gz"
    write_filter_report(report, report_fp)
    print_filter_summary(report, args.qtl)
    print(f"Saved QTL filter report to {report_fp}")
    keep = eligible_snps(report)
    expt = {snp: expt[snp] for snp in keep}
    snps = list(expt)
    ref = pd.DataFrame({snp: expt[snp][0].mean(axis=0) for snp in snps}).transpose()
    alt = pd.DataFrame({snp: expt[snp][1].mean(axis=0) for snp in snps}).transpose()
    return expt, ref, alt, report


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


def load_or_calculate_scores(
    args,
    out_fp,
    pred_fp,
    snps,
    expt_ref,
    expt_alt,
    qtl_coord,
    fold=None,
):
    if args.skip_existing and out_fp.exists():
        scores = pd.read_csv(out_fp, index_col=0)
        return scores.loc[[snp for snp in snps if snp in scores.index]]

    import joblib

    pred = joblib.load(pred_fp)[0]
    score_snps = [
        snp
        for snp in snps
        if snp in pred
        and pred[snp][0] is not None
        and pred[snp][1] is not None
        and np.asarray(pred[snp][0]).size
        and np.asarray(pred[snp][1]).size
    ]
    if len(score_snps) != len(snps):
        logging.warning(
            "Dropping %d eligible SNPs without both predicted homozygous groups in %s.",
            len(snps) - len(score_snps),
            pred_fp,
        )
    if not score_snps:
        raise ValueError(f"No eligible SNP predictions remain in {pred_fp}.")
    pred_ref = pd.DataFrame(
        {snp: np.mean(pred[snp][0], axis=0) for snp in score_snps}
    ).transpose()
    pred_alt = pd.DataFrame(
        {snp: np.mean(pred[snp][1], axis=0) for snp in score_snps}
    ).transpose()
    scores = pd.DataFrame(
        {
            "expt": l2_score(
                expt_ref.loc[score_snps].to_numpy(),
                expt_alt.loc[score_snps].to_numpy(),
            ),
            "pred": l2_score(pred_ref.to_numpy(), pred_alt.to_numpy()),
        },
        index=score_snps,
    )
    scores["expt_log_l2"] = log_l2_values(scores["expt"].to_numpy())
    scores["pred_log_l2"] = log_l2_values(scores["pred"].to_numpy())
    qtl_index = qtl_coord.set_index("snps")
    scores["chrom"] = qtl_index.loc[scores.index, "chrom"]
    scores["start"] = qtl_index.loc[scores.index, "start"]
    if fold is not None:
        keep = holdout_snps(args, fold, qtl_coord)
        scores = scores.loc[[snp for snp in scores.index if snp in keep]]
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_fp)
    return scores


def score_best_model(args, snps, expt_ref, expt_alt, qtl_coord, fold=None):
    out_fp = score_path(args) if fold is None else fold_score_path(args, fold)
    pred_fp = split_path(args) if fold is None else fold_split_path(args, fold)
    scores = load_or_calculate_scores(
        args, out_fp, pred_fp, snps, expt_ref, expt_alt, qtl_coord, fold=fold
    )
    aggregation = "ensemble" if fold is None else "fold"
    return summarize_scores(args, scores, np.nan if fold is None else fold, aggregation)


def load_ensemble_scores(args, snps, expt_ref, expt_alt, qtl_coord):
    out_fp = ensemble_score_path(args)
    pred_fp = ensemble_split_path(args)
    if out_fp.exists():
        scores = pd.read_csv(out_fp, index_col=0)
        return scores.loc[[snp for snp in snps if snp in scores.index]]
    elif pred_fp.exists():
        return load_or_calculate_scores(
            args, out_fp, pred_fp, snps, expt_ref, expt_alt, qtl_coord
        )
    print(
        "Skipping legacy composite aggregation: "
        f"missing {out_fp} and {pred_fp}."
    )
    return None


def score_fold_best_pooled(args, folds, snps):
    fold_scores = []
    for fold in folds:
        if str(fold) == "0":
            continue
        fold_fp = fold_score_path(args, fold)
        if fold_fp.exists():
            fold_score = pd.read_csv(fold_fp, index_col=0)
            fold_score = fold_score.loc[
                [snp for snp in snps if snp in fold_score.index]
            ]
            fold_score["fold"] = fold
            fold_scores.append(fold_score)
    if not fold_scores:
        return None
    pooled_scores = pd.concat(fold_scores)
    return summarize_scores(args, pooled_scores, "pooled", "pooled_folds")


def score_fold_best_pooled_with_fold0_ensemble(
    args, folds, snps, expt_ref, expt_alt, qtl_coord
):
    fold_scores = []
    for fold in folds:
        if str(fold) == "0":
            continue
        fold_fp = fold_score_path(args, fold)
        if fold_fp.exists():
            fold_score = pd.read_csv(fold_fp, index_col=0)
            fold_score = fold_score.loc[
                [snp for snp in snps if snp in fold_score.index]
            ]
            fold_score["fold"] = fold
            fold_scores.append(fold_score)

    ensemble_scores = load_ensemble_scores(
        args, snps, expt_ref, expt_alt, qtl_coord
    )
    if not fold_scores or ensemble_scores is None:
        return None
    fold_scores = pd.concat(fold_scores)
    ensemble_remainder = ensemble_scores.loc[
        ~ensemble_scores.index.isin(fold_scores.index)
    ].copy()
    ensemble_remainder["fold"] = "ensemble_remainder"
    pooled_scores = pd.concat([fold_scores, ensemble_remainder])
    return summarize_scores(
        args,
        pooled_scores,
        "legacy_composite",
        "legacy_composite",
    )


def run_score(args):
    expt, expt_ref, expt_alt, _ = load_experimental_tracks(args)
    snps = list(expt)
    qtl_coord = load_qtl_coordinates(args, snps)
    checkpoints = discover_checkpoints(args)
    folds = selected_folds(args, checkpoints)

    summary_rows = []
    if args.mode == "ensemble":
        summary_rows.append(score_best_model(args, snps, expt_ref, expt_alt, qtl_coord))
    else:
        checkpoint_paths(args, checkpoints, folds)
        for fold in tqdm.tqdm(folds, desc="Scoring best-model QTL predictions"):
            if fold in checkpoints:
                summary_rows.append(
                    score_best_model(args, snps, expt_ref, expt_alt, qtl_coord, fold=fold)
                )
        pooled_row = score_fold_best_pooled(args, folds, snps)
        if pooled_row is not None:
            summary_rows.append(pooled_row)
        pooled_with_fold0_row = score_fold_best_pooled_with_fold0_ensemble(
            args, folds, snps, expt_ref, expt_alt, qtl_coord
        )
        if pooled_with_fold0_row is not None:
            summary_rows.append(pooled_with_fold0_row)

    out_fp = summary_path(args)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).sort_values(["aggregation", "fold"]).to_csv(out_fp, index=False)
    print(f"Saved best-model benchmark summary to {out_fp}")


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
