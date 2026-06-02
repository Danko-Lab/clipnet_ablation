#!/usr/bin/env python3

"""
Consolidated CLIPNET QTL prediction workflow.

This replaces the duplicated diQTL/tiQTL fold and ensemble scripts with one
entrypoint for prediction, allele grouping, and L2 score calculation.
"""

import argparse
import itertools
import json
import logging
import os
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
        "--qtl",
        choices=sorted(QTL_CONFIG),
        required=True,
        help="QTL dataset to evaluate.",
    )
    parser.add_argument(
        "--mode",
        choices=["ensemble", "folds"],
        required=True,
        help="Use ensemble predictions or per-fold predictions.",
    )
    parser.add_argument(
        "--n_individuals",
        type=parse_ints,
        default=parse_ints("5,10,15,20,30"),
        help="Comma-separated subsample sizes. Default: 5,10,15,20,30.",
    )
    parser.add_argument(
        "--runs",
        type=parse_ints,
        default=parse_ints("0,1,2,3,4"),
        help="Comma-separated run IDs. Default: 0,1,2,3,4.",
    )
    parser.add_argument(
        "--folds",
        type=parse_ints,
        default=parse_ints("1,2,3,4,5,6,7,8,9"),
        help="Comma-separated folds for --mode folds. Default: 1..9.",
    )
    parser.add_argument(
        "--models_root",
        type=Path,
        default=REPO_ROOT / "models",
        help="Root containing n{N}_run{R} model directories.",
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
        help="Root where predictions and score outputs are stored.",
    )
    parser.add_argument(
        "--qtl_data_dir",
        type=Path,
        default=None,
        help="Directory containing allele matrix, experimental tracks, and QTL table.",
    )
    parser.add_argument(
        "--prefix_map",
        type=Path,
        default=REPO_ROOT / "data_spec/procap_to_1k_genomes.json",
        help="JSON map used to discover PRO-cap prefixes for prediction.",
    )
    parser.add_argument(
        "--missing_prefixes",
        type=Path,
        default=REPO_ROOT / "data_spec/missing_1k_genomes.txt",
        help="Text file of missing sequence prefixes to skip.",
    )
    parser.add_argument(
        "--fold_assignments",
        type=Path,
        default=REPO_ROOT / "clipnet_data_folds/data_fold_assignments.csv",
        help="Fold assignment CSV used to hold out SNPs in fold score mode.",
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
        "--prefixes",
        type=str,
        default=None,
        help="Optional comma-separated PRO-cap prefixes to process.",
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


def qtl_prediction_root(args):
    return args.predictions_root / args.qtl


def mode_dir(mode):
    return "ensemble_predict" if mode == "ensemble" else "fold_predict"


def run_name(n_individuals, run):
    return f"n{n_individuals}_run{run}"


def load_prefixes(args):
    if args.prefixes is not None:
        return [prefix for prefix in args.prefixes.split(",") if prefix.strip()]

    with open(args.prefix_map, "r") as handle:
        prefix_map = json.load(handle)
    with open(args.missing_prefixes, "r") as handle:
        missing = set(handle.read().splitlines())
    return [prefix for prefix, seq_prefix in prefix_map.items() if seq_prefix not in missing]


def prediction_path(args, run, prefix, fold=None):
    root = qtl_prediction_root(args) / mode_dir(args.mode)
    if args.mode == "ensemble":
        return root / run / f"{prefix}.h5"
    return root / run / f"fold_{fold}" / f"{prefix}.h5"


def split_path(args, run, fold=None):
    root = qtl_prediction_root(args) / mode_dir(args.mode) / "split_by_allele"
    if args.mode == "ensemble":
        return root / f"{run}_pred_per_snp_by_allele.joblib.gz"
    return root / f"{run}_fold_{fold}_pred_per_snp_by_allele.joblib.gz"


def score_path(args, run, fold=None):
    root = qtl_prediction_root(args) / mode_dir(args.mode) / "split_by_allele"
    if args.mode == "ensemble":
        return root / f"{run}_l2_scores.csv.gz"
    return root / f"{run}_fold_{fold}_l2_scores.csv.gz"


def resolve_fold_model(model_dir, fold):
    candidates = [
        model_dir / f"fold_{fold}.h5",
        model_dir / f"fold_{fold}.hdf5",
        model_dir / f"f{fold}.h5",
        model_dir / f"f{fold}.hdf5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No fold {fold} model found in {model_dir}; checked {candidates}."
    )


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
    return tf


def create_predictor(args):
    if args.dry_run:
        return None
    configure_prediction_runtime(args)
    from clipnet import clipnet

    return clipnet.CLIPNET(
        n_gpus=1 if args.gpu else 0,
        use_specific_gpu=args.use_specific_gpu if args.gpu else -1,
    )


def predict_one(nn, model_fp, sequence_fp, output_fp, args):
    if args.dry_run:
        print(f"predict {model_fp} {sequence_fp} -> {output_fp}")
        return
    if args.skip_existing and output_fp.exists():
        return

    output_fp.parent.mkdir(parents=True, exist_ok=True)
    import h5py

    prediction = nn.predict_on_fasta(
        model_fp=str(model_fp),
        fasta_fp=str(sequence_fp),
        low_mem=True,
        silence=True,
    )
    with h5py.File(output_fp, "w") as hf:
        hf.create_dataset("track", data=prediction[0], compression=args.compression)
        hf.create_dataset("quantity", data=prediction[1], compression=args.compression)


def run_predict(args):
    prefixes = load_prefixes(args)
    sequence_root = args.data_root / args.qtl / "sequence"
    nn = create_predictor(args)

    for n_individuals, run_id in itertools.product(args.n_individuals, args.runs):
        run = run_name(n_individuals, run_id)
        model_dir = args.models_root / run
        if args.mode == "ensemble":
            jobs = [(model_dir, None)]
        else:
            if args.dry_run:
                jobs = [(model_dir / f"fold_{fold}.h5", fold) for fold in args.folds]
            else:
                jobs = [(resolve_fold_model(model_dir, fold), fold) for fold in args.folds]

        for model_fp, fold in jobs:
            for prefix in prefixes:
                sequence_fp = sequence_root / f"{prefix}.fna.gz"
                output_fp = prediction_path(args, run, prefix, fold=fold)
                predict_one(nn, model_fp, sequence_fp, output_fp, args)


def load_allele_matrix(args):
    config = QTL_CONFIG[args.qtl]
    allele_fp = qtl_data_dir(args) / config["alleles"]
    return pd.read_csv(allele_fp, index_col=0)


def scaled_prediction(pred_fp):
    import h5py

    with h5py.File(pred_fp, "r") as pred:
        track = pred["track"][:]
        quantity = pred["quantity"][:]
    if quantity.ndim == 2 and quantity.shape[1] == 1:
        quantity = quantity[:, 0]
    return (track / (track.sum(axis=1, keepdims=True) + 1e-3)) * quantity[:, None], quantity


def split_prediction_set(args, run, allele_matrix, prefixes, snps, fold=None):
    out_fp = split_path(args, run, fold=fold)
    if args.skip_existing and out_fp.exists():
        return

    pred_tracks = {snp: [[], [], []] for snp in snps}
    pred_quantities = {snp: [[], [], []] for snp in snps}

    for prefix in prefixes:
        pred_fp = prediction_path(args, run, prefix, fold=fold)
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
    prefixes = list(allele_matrix.index)
    snps = list(allele_matrix.columns)
    runs = [
        run_name(n_individuals, run_id)
        for n_individuals, run_id in itertools.product(args.n_individuals, args.runs)
    ]

    for run in tqdm.tqdm(runs, desc="Splitting predictions by allele"):
        if args.mode == "ensemble":
            split_prediction_set(args, run, allele_matrix, prefixes, snps)
        else:
            for fold in args.folds:
                split_prediction_set(args, run, allele_matrix, prefixes, snps, fold=fold)


def l2_score(x, y):
    return np.sqrt(np.sum(np.square(x - y), axis=1).astype(np.float32))


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


def score_prediction_set(args, run, snps, expt_ref, expt_alt, qtl_coord, fold=None):
    out_fp = score_path(args, run, fold=fold)
    if args.skip_existing and out_fp.exists():
        return

    import joblib

    pred = joblib.load(split_path(args, run, fold=fold))[0]
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
    if fold is not None:
        keep = holdout_snps(args, fold, qtl_coord)
        scores = scores.loc[[snp for snp in scores.index if snp in keep]]

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_fp)


def run_score(args):
    expt, expt_ref, expt_alt = load_experimental_tracks(args)
    snps = list(expt)
    qtl_coord = load_qtl_coordinates(args, snps)
    runs = [
        run_name(n_individuals, run_id)
        for n_individuals, run_id in itertools.product(args.n_individuals, args.runs)
    ]

    for run in tqdm.tqdm(runs, desc="Calculating QTL scores"):
        if args.mode == "ensemble":
            score_prediction_set(args, run, snps, expt_ref, expt_alt, qtl_coord)
        else:
            for fold in args.folds:
                score_prediction_set(
                    args, run, snps, expt_ref, expt_alt, qtl_coord, fold=fold
                )


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
