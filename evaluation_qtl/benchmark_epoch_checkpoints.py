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

try:
    from qtl_filters import (
        build_filter_report,
        eligible_snps,
        load_prefix_map,
        print_filter_summary,
        write_filter_report,
    )
    from published_qtl_targets import (
        experimental_l2_targets,
        experimental_source,
        score_directory_name,
        summary_suffix,
    )
    from fold_calibrated_scoring import build_fold_calibrated_rows
except ImportError:
    from evaluation_qtl.qtl_filters import (
        build_filter_report,
        eligible_snps,
        load_prefix_map,
        print_filter_summary,
        write_filter_report,
    )
    from evaluation_qtl.published_qtl_targets import (
        experimental_l2_targets,
        experimental_source,
        score_directory_name,
        summary_suffix,
    )
    from evaluation_qtl.fold_calibrated_scoring import build_fold_calibrated_rows


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
SEQUENCE_COORD_RE = re.compile(r"(chr[^:|,\s_]+):(\d+)-(\d+)")
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
        "--experimental_l2_archive",
        type=Path,
        default=None,
        help=(
            "Use canonical per-SNP observed L2 values from the official "
            "qtl_analysis.tar.gz instead of generated experimental tracks."
        ),
    )
    parser.add_argument(
        "--qtl_table",
        type=Path,
        default=None,
        help="QTL coordinate table with snps and gene columns.",
    )
    parser.add_argument(
        "--qtl_snp_bed",
        type=Path,
        default=None,
        help=(
            "BED(.gz) containing QTL SNP coordinates and rsIDs. Defaults to "
            "{qtl_data_dir}/diQTL_snps.bed.gz or tiQTL_snps.bed.gz."
        ),
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


def resolve_qtl_snp_bed(args):
    if getattr(args, "qtl_snp_bed", None) is not None:
        return args.qtl_snp_bed
    qdir = qtl_data_dir(args)
    stem = "diQTL" if args.qtl == "diqtl" else "tiQTL"
    candidates = [
        qdir / f"{stem}_snps.bed.gz",
        qdir / f"{stem}_snps.bed",
        qdir / f"{args.qtl}_snps.bed.gz",
        qdir / f"{args.qtl}_snps.bed",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_snp_coordinate_map(args, snps):
    snp_bed = resolve_qtl_snp_bed(args)
    if snp_bed is not None:
        bed = pd.read_csv(snp_bed, sep="\t", header=None, comment="#")
        snp_set = set(snps)
        coordinate_to_snps = {}
        found = set()
        for row in bed.itertuples(index=False, name=None):
            if len(row) < 4:
                continue
            snp_matches = []
            for value in row[3:]:
                text = str(value)
                candidates = [text, *re.findall(r"rs\d+", text)]
                snp_matches.extend(snp for snp in candidates if snp in snp_set)
            snp_matches = list(dict.fromkeys(snp_matches))
            if not snp_matches:
                continue
            chrom, start, end = str(row[0]), int(row[1]), int(row[2])
            for position in {start, end, start + 1, end - 1}:
                key = (chrom, position)
                values = coordinate_to_snps.setdefault(key, [])
                for snp in snp_matches:
                    if snp not in values:
                        values.append(snp)
                    found.add(snp)
        missing = [snp for snp in snps if snp not in found]
        if missing:
            raise ValueError(
                f"{snp_bed} does not provide coordinates for allele-matrix SNPs: "
                f"{missing[:5]} ({len(missing)} total)."
            )
        return coordinate_to_snps

    logging.warning(
        "No QTL SNP BED found under %s; falling back to QTL-table gene "
        "coordinates, which may not represent variant positions.",
        qtl_data_dir(args),
    )
    config = QTL_CONFIG[args.qtl]
    table_fp = args.qtl_table or qtl_data_dir(args) / config["table"]
    table = pd.read_csv(table_fp, sep=config["table_sep"]).drop_duplicates(
        subset="snps", keep="first"
    )
    table = table[table["snps"].isin(snps)]
    coordinates = table["gene"].astype(str).str.rsplit(
        config["gene_sep"], n=1, expand=True
    )
    if coordinates.shape[1] != 2:
        raise ValueError(f"Could not parse QTL coordinates from {table_fp}.")

    snp_to_coordinate = {}
    for snp, chrom, position in zip(table["snps"], coordinates[0], coordinates[1]):
        try:
            key = (chrom, int(position))
        except (TypeError, ValueError):
            continue
        snp_to_coordinate[snp] = key

    coordinate_to_snps = {}
    for snp in snps:
        coordinate = snp_to_coordinate.get(snp)
        if coordinate is not None:
            coordinate_to_snps.setdefault(coordinate, []).append(snp)
    return coordinate_to_snps


def match_sequence_coordinate(sequence_id, coordinate_to_snps, seen):
    matches = []
    for chrom, start, end in SEQUENCE_COORD_RE.findall(sequence_id):
        start = int(start)
        end = int(end)
        for position in {(start + end) // 2, (start + end + 1) // 2}:
            matches.extend(
                snp
                for snp in coordinate_to_snps.get((chrom, position), [])
                if snp not in seen
            )
    matches = list(dict.fromkeys(matches))
    if matches:
        return matches[0]
    return None


def prediction_row_order(
    pred_fp, snps, coordinate_to_snp=None, allow_row_order=False
):
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
        if snp is None and coordinate_to_snp is not None:
            snp = match_sequence_coordinate(sequence_id, coordinate_to_snp, seen)
        if snp is None:
            unmatched.append((row_idx, sequence_id))
            continue
        if snp in seen:
            raise ValueError(
                f"Duplicate sequence records map to SNP {snp} in {pred_fp}."
            )
        seen.add(snp)
        rows.append((row_idx, snp))

    missing = [snp for snp in snps if snp not in seen]
    if unmatched or missing:
        if len(unmatched) == len(missing) == 1:
            row_idx, sequence_id = unmatched[0]
            logging.warning(
                "Uniquely mapping unmatched sequence %s to remaining SNP %s in %s.",
                sequence_id,
                missing[0],
                pred_fp,
            )
            rows.append((row_idx, missing[0]))
            return sorted(rows)
        if allow_row_order and len(unmatched) == len(missing):
            logging.warning(
                "Using unsafe row-order mapping for %d unmatched prediction rows "
                "and SNPs in %s because --allow_row_order was set.",
                len(unmatched),
                pred_fp,
            )
            rows.extend(
                (row_idx, snp)
                for (row_idx, _sequence_id), snp in zip(unmatched, missing)
            )
            return sorted(rows)
        unmatched_ids = [sequence_id for _, sequence_id in unmatched]
        raise ValueError(
            f"Prediction sequence IDs in {pred_fp} do not match the allele matrix. "
            f"Unmatched sequence IDs: {unmatched_ids[:5]} "
            f"({len(unmatched)} total). Missing SNPs: {missing[:5]} "
            f"({len(missing)} total). To map only the unmatched remainder by "
            f"allele-matrix order, rerun with --allow_row_order."
        )
    return rows


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


def selected_epochs(args, checkpoints, folds):
    if args.epochs is None:
        epochs = sorted(checkpoints)
        if not args.allow_missing_folds:
            complete_epochs = [
                epoch
                for epoch in epochs
                if all(fold in checkpoints[epoch] for fold in folds)
            ]
            skipped = sorted(set(epochs) - set(complete_epochs))
            if skipped:
                print(
                    "Skipping epochs missing one or more selected folds: "
                    f"{skipped}. Use --allow_missing_folds to evaluate them."
                )
            epochs = complete_epochs
            if not epochs:
                raise FileNotFoundError(
                    "No epochs have checkpoints for all selected folds. "
                    "Use --allow_missing_folds to evaluate available folds."
                )
    else:
        epochs = args.epochs
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


def ensemble_split_path(args, epoch):
    return (
        benchmark_root(args)
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
        / score_directory_name(args)
        / f"epoch_{epoch:03d}_l2_scores.csv.gz"
    )


def ensemble_score_path(args, epoch):
    return (
        benchmark_root(args)
        / score_directory_name(args)
        / f"epoch_{epoch:03d}_l2_scores.csv.gz"
    )


def fold_score_path(args, epoch, fold):
    return (
        output_root(args)
        / score_directory_name(args)
        / f"epoch_{epoch:03d}_fold_{fold}_l2_scores.csv.gz"
    )


def summary_path(args):
    return (
        output_root(args)
        / f"epoch_qtl_benchmark_summary{summary_suffix(args)}.csv"
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
    epochs = selected_epochs(args, checkpoints, folds)
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
            sequence_ids = None if args.dry_run else fasta_sequence_ids(sequence_fp)
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


def split_epoch_predictions(
    args, epoch, allele_matrix, prefixes, snps, coordinate_to_snp, fold=None
):
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
        for row_idx, snp in prediction_row_order(
            pred_fp,
            snps,
            coordinate_to_snp=coordinate_to_snp,
            allow_row_order=args.allow_row_order,
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
    coordinate_to_snp = load_snp_coordinate_map(args, snps)
    checkpoints = discover_checkpoints(args)
    folds = selected_folds(args, checkpoints)
    epochs = selected_epochs(args, checkpoints, folds)

    for epoch in tqdm.tqdm(epochs, desc="Splitting epoch predictions by allele"):
        if args.mode == "ensemble":
            split_epoch_predictions(
                args,
                epoch,
                allele_matrix,
                prefixes,
                snps,
                coordinate_to_snp,
            )
        else:
            epoch_checkpoint_paths(args, checkpoints, epoch, folds)
            for fold in folds:
                if fold in checkpoints[epoch]:
                    split_epoch_predictions(
                        args,
                        epoch,
                        allele_matrix,
                        prefixes,
                        snps,
                        coordinate_to_snp,
                        fold=fold,
                    )


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


def summarize_scores(args, epoch, scores, fold, aggregation):
    expt_values = scores["expt"].to_numpy()
    pred_values = scores["pred"].to_numpy()
    expt_log_values = log_l2_values(expt_values)
    pred_log_values = log_l2_values(pred_values)
    finite = np.isfinite(expt_log_values) & np.isfinite(pred_log_values)

    return {
        "mode": args.mode,
        "aggregation": aggregation,
        "epoch": epoch,
        "experimental_source": experimental_source(args),
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
    targets = experimental_l2_targets(args)
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
            "expt": (
                targets.loc[score_snps].to_numpy()
                if targets is not None
                else l2_score(
                    expt_ref.loc[score_snps].to_numpy(),
                    expt_alt.loc[score_snps].to_numpy(),
                )
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


def score_epoch(args, epoch, snps, expt_ref, expt_alt, qtl_coord, fold=None):
    out_fp = score_path(args, epoch) if fold is None else fold_score_path(args, epoch, fold)
    pred_fp = split_path(args, epoch) if fold is None else fold_split_path(args, epoch, fold)
    scores = load_or_calculate_scores(
        args, out_fp, pred_fp, snps, expt_ref, expt_alt, qtl_coord, fold=fold
    )
    aggregation = "ensemble" if fold is None else "fold"
    return summarize_scores(args, epoch, scores, np.nan if fold is None else fold, aggregation)


def load_ensemble_scores(args, epoch, snps, expt_ref, expt_alt, qtl_coord):
    out_fp = ensemble_score_path(args, epoch)
    pred_fp = ensemble_split_path(args, epoch)
    if out_fp.exists():
        scores = pd.read_csv(out_fp, index_col=0)
        return scores.loc[[snp for snp in snps if snp in scores.index]]
    elif pred_fp.exists():
        return load_or_calculate_scores(
            args, out_fp, pred_fp, snps, expt_ref, expt_alt, qtl_coord
        )
    print(
        "Skipping legacy composite aggregation for "
        f"epoch {epoch}: missing {out_fp} and {pred_fp}."
    )
    return None


def load_fold_epoch_scores(args, epoch, folds, snps):
    fold_scores = {}
    for fold in folds:
        if str(fold) == "0":
            continue
        fold_fp = fold_score_path(args, epoch, fold)
        if fold_fp.exists():
            fold_score = pd.read_csv(fold_fp, index_col=0)
            fold_score = fold_score.loc[
                [snp for snp in snps if snp in fold_score.index]
            ]
            fold_score["fold"] = fold
            fold_scores[fold] = fold_score
    return fold_scores


def best_fold_epoch_scores(args, epochs, folds, snps):
    best_scores = {}
    best_rows = []
    for fold in folds:
        if str(fold) == "0":
            continue
        candidates = []
        for epoch in epochs:
            fold_fp = fold_score_path(args, epoch, fold)
            if not fold_fp.exists():
                continue
            fold_score = pd.read_csv(fold_fp, index_col=0)
            fold_score = fold_score.loc[
                [snp for snp in snps if snp in fold_score.index]
            ].copy()
            if fold_score.empty:
                continue
            row = summarize_scores(args, epoch, fold_score, fold, "fold")
            candidates.append((row["log_l2_pearson"], epoch, fold_score, row))
        finite_candidates = [
            candidate
            for candidate in candidates
            if np.isfinite(candidate[0])
        ]
        if not finite_candidates:
            continue
        _, selected_epoch, selected_score, selected_row = max(
            finite_candidates, key=lambda candidate: candidate[0]
        )
        selected_score["fold"] = fold
        selected_score["selected_epoch"] = selected_epoch
        best_scores[fold] = selected_score
        selected_row = dict(selected_row)
        selected_row["epoch"] = np.nan
        selected_row["selected_epoch"] = selected_epoch
        selected_row["aggregation"] = "best_per_fold"
        best_rows.append(selected_row)
    return best_scores, best_rows


def score_fold_epoch_pooled(args, epoch, fold_scores):
    if not fold_scores:
        return None
    pooled_scores = pd.concat(fold_scores.values())
    return summarize_scores(args, epoch, pooled_scores, "pooled", "pooled_folds")


def score_fold0_epoch_ensemble(
    args, epoch, fold_scores, snps, expt_ref, expt_alt, qtl_coord
):
    ensemble_scores = load_ensemble_scores(
        args, epoch, snps, expt_ref, expt_alt, qtl_coord
    )
    if not fold_scores or ensemble_scores is None:
        return None
    fold_scores = pd.concat(fold_scores.values())
    ensemble_remainder = ensemble_scores.loc[
        ~ensemble_scores.index.isin(fold_scores.index)
    ].copy()
    if ensemble_remainder.empty:
        return None
    ensemble_remainder["fold"] = "fold0_ensemble"
    return summarize_scores(
        args,
        epoch,
        ensemble_remainder,
        "fold0_ensemble",
        "fold0_ensemble",
    )


def score_best_per_fold_pooled(args, fold_scores):
    if not fold_scores:
        return None
    pooled_scores = pd.concat(fold_scores.values())
    row = summarize_scores(
        args, np.nan, pooled_scores, "best_per_fold_pooled", "best_per_fold_pooled"
    )
    row["selected_epoch"] = selected_epoch_summary(fold_scores)
    return row


def selected_epoch_summary(fold_scores):
    selected = {
        str(fold): int(scores["selected_epoch"].iloc[0])
        for fold, scores in fold_scores.items()
        if "selected_epoch" in scores
    }
    return ",".join(f"{fold}:{epoch}" for fold, epoch in sorted(selected.items()))


def score_best_per_fold_fold0_ensemble(
    args, epochs, fold_scores, snps, expt_ref, expt_alt, qtl_coord
):
    if not fold_scores:
        return None
    heldout_scores = pd.concat(fold_scores.values())
    candidates = []
    for epoch in epochs:
        ensemble_scores = load_ensemble_scores(
            args, epoch, snps, expt_ref, expt_alt, qtl_coord
        )
        if ensemble_scores is None:
            continue
        ensemble_remainder = ensemble_scores.loc[
            ~ensemble_scores.index.isin(heldout_scores.index)
        ].copy()
        if ensemble_remainder.empty:
            continue
        row = summarize_scores(
            args,
            epoch,
            ensemble_remainder,
            "fold0_ensemble",
            "fold0_ensemble",
        )
        candidates.append((row["log_l2_pearson"], epoch, ensemble_remainder, row))
    finite_candidates = [
        candidate for candidate in candidates if np.isfinite(candidate[0])
    ]
    if not finite_candidates:
        return None
    _, selected_epoch, selected_scores, row = max(
        finite_candidates, key=lambda candidate: candidate[0]
    )
    selected_scores["fold"] = "fold0_ensemble"
    selected_scores["selected_epoch"] = selected_epoch
    row = dict(row)
    row["epoch"] = np.nan
    row["selected_epoch"] = selected_epoch
    row["aggregation"] = "best_per_fold_fold0_ensemble"
    return row, selected_scores


def score_best_per_fold_legacy_composite(
    args, fold_scores, fold0_scores
):
    if not fold_scores or fold0_scores is None:
        return None
    pooled_scores = pd.concat([pd.concat(fold_scores.values()), fold0_scores])
    row = summarize_scores(
        args,
        np.nan,
        pooled_scores,
        "best_per_fold_legacy_composite",
        "best_per_fold_legacy_composite",
    )
    selected = selected_epoch_summary(fold_scores)
    if "selected_epoch" in fold0_scores and not fold0_scores.empty:
        fold0_epoch = int(fold0_scores["selected_epoch"].iloc[0])
        selected = f"{selected},fold0:{fold0_epoch}" if selected else f"fold0:{fold0_epoch}"
    row["selected_epoch"] = selected
    return row


def score_fold_epoch_pooled_with_fold0_ensemble(
    args, epoch, fold_scores, snps, expt_ref, expt_alt, qtl_coord
):
    ensemble_scores = load_ensemble_scores(
        args, epoch, snps, expt_ref, expt_alt, qtl_coord
    )
    if not fold_scores or ensemble_scores is None:
        return None
    fold_scores = pd.concat(fold_scores.values())
    ensemble_remainder = ensemble_scores.loc[
        ~ensemble_scores.index.isin(fold_scores.index)
    ].copy()
    ensemble_remainder["fold"] = "ensemble_remainder"
    pooled_scores = pd.concat([fold_scores, ensemble_remainder])
    return summarize_scores(
        args,
        epoch,
        pooled_scores,
        "legacy_composite",
        "legacy_composite",
    )


def run_score(args):
    targets = experimental_l2_targets(args)
    if targets is None:
        expt, expt_ref, expt_alt, _ = load_experimental_tracks(args)
        snps = list(expt)
    else:
        expt_ref = expt_alt = None
        snps = list(targets.index)
        print(
            f"Using {len(snps):,} canonical observed L2 targets from "
            f"{args.experimental_l2_archive}."
        )
    qtl_coord = load_qtl_coordinates(args, snps)
    checkpoints = discover_checkpoints(args)
    folds = selected_folds(args, checkpoints)
    epochs = selected_epochs(args, checkpoints, folds)

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
            fold_scores = load_fold_epoch_scores(args, epoch, folds, snps)
            pooled_row = score_fold_epoch_pooled(args, epoch, fold_scores)
            if pooled_row is not None:
                summary_rows.append(pooled_row)
                summary_rows.extend(
                    build_fold_calibrated_rows(
                        fold_scores,
                        lambda scores, fold, aggregation: summarize_scores(
                            args, epoch, scores, fold, aggregation
                        ),
                        log_l2_values,
                        correlation,
                        pooled_row,
                    )
                )
            pooled_with_fold0_row = score_fold_epoch_pooled_with_fold0_ensemble(
                args, epoch, fold_scores, snps, expt_ref, expt_alt, qtl_coord
            )
            fold0_ensemble_row = score_fold0_epoch_ensemble(
                args, epoch, fold_scores, snps, expt_ref, expt_alt, qtl_coord
            )
            if pooled_with_fold0_row is not None:
                summary_rows.append(pooled_with_fold0_row)
            if fold0_ensemble_row is not None:
                summary_rows.append(fold0_ensemble_row)

    if args.mode == "folds":
        best_scores, best_rows = best_fold_epoch_scores(args, epochs, folds, snps)
        summary_rows.extend(best_rows)
        best_pooled_row = score_best_per_fold_pooled(args, best_scores)
        if best_pooled_row is not None:
            summary_rows.append(best_pooled_row)
            summary_rows.extend(
                build_fold_calibrated_rows(
                    best_scores,
                    lambda scores, fold, aggregation: summarize_scores(
                        args, np.nan, scores, fold, f"best_per_fold_{aggregation}"
                    ),
                    log_l2_values,
                    correlation,
                    best_pooled_row,
                )
            )
        fold0_result = score_best_per_fold_fold0_ensemble(
            args, epochs, best_scores, snps, expt_ref, expt_alt, qtl_coord
        )
        if fold0_result is not None:
            fold0_row, fold0_scores = fold0_result
            summary_rows.append(fold0_row)
            best_legacy_row = score_best_per_fold_legacy_composite(
                args, best_scores, fold0_scores
            )
            if best_legacy_row is not None:
                summary_rows.append(best_legacy_row)

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
