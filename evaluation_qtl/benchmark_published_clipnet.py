#!/usr/bin/env python3

"""
Benchmark the published CLIPNET model folds from Zenodo record 10408623.

Download fold_1.h5 through fold_9.h5 and data_fold_assignments.csv into one
model directory, then use this entrypoint to run the same prediction, allele
grouping, manuscript filtering, and QTL scoring pipeline used for the ablation
models.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import benchmark_best_model as benchmark
except ImportError:
    from evaluation_qtl import benchmark_best_model as benchmark


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PUBLISHED_EXPECTED_PEARSON = {"tiqtl": 0.477, "diqtl": 0.542}
PUBLISHED_FOLD_RE = re.compile(r"^fold_(\d+)\.h(?:5|df5)$")


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
        help="Directory containing the downloaded fold_1.h5 ... fold_9.h5 files.",
    )
    parser.add_argument("--qtl", choices=["diqtl", "tiqtl"], required=True)
    parser.add_argument(
        "--run_name",
        default="zenodo_10408623",
        help="Output run name.",
    )
    parser.add_argument(
        "--mode",
        choices=["composite", "ensemble", "folds"],
        default="composite",
        help=(
            "Inference mode. Composite runs ensemble and folds in sequence and "
            "produces the manuscript-style held-out-fold plus fold-0 ensemble result."
        ),
    )
    parser.add_argument(
        "--folds",
        type=parse_ints,
        default=parse_ints("1,2,3,4,5,6,7,8,9"),
        help="Comma-separated folds. Default: 1..9.",
    )
    parser.add_argument("--allow_missing_folds", action="store_true")
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
    )
    parser.add_argument("--qtl_data_dir", type=Path, default=None)
    parser.add_argument("--expt_by_allele", type=Path, default=None)
    parser.add_argument("--qtl_table", type=Path, default=None)
    parser.add_argument(
        "--prefix_map",
        type=Path,
        default=REPO_ROOT / "data_spec/procap_to_1k_genomes.json",
    )
    parser.add_argument("--pvalue_column", default=None)
    parser.add_argument(
        "--fold_assignments",
        type=Path,
        default=None,
        help=(
            "Chromosome fold assignments. Defaults to data_fold_assignments.csv "
            "inside model_root, then the repository assignment file."
        ),
    )
    parser.add_argument("--prefixes", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--use_specific_gpu", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--allow_row_order",
        action="store_true",
        help="Allow coordinate FASTA rows to map to allele-matrix SNPs by exact row order.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--compression", default="gzip")
    return parser.parse_args()


def discover_published_checkpoints(args):
    model_root = args.model_root.resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"Published model root does not exist: {model_root}")

    checkpoints = {}
    for path in sorted(model_root.iterdir()):
        match = PUBLISHED_FOLD_RE.match(path.name)
        if match is None:
            continue
        fold = int(match.group(1))
        if fold in checkpoints:
            raise ValueError(
                f"Multiple published checkpoints found for fold {fold}: "
                f"{checkpoints[fold]} and {path}"
            )
        checkpoints[fold] = path

    if not checkpoints:
        raise FileNotFoundError(
            f"No published fold_N.h5/.hdf5 checkpoints found in {model_root}."
        )
    unexpected = sorted(set(checkpoints) - set(range(1, 10)))
    if unexpected:
        raise ValueError(f"Unexpected published fold IDs: {unexpected}")
    return checkpoints


def benchmark_root(args):
    return (
        args.predictions_root
        / args.qtl
        / "published_clipnet_benchmark"
        / args.run_name
    )


def prediction_path(args, prefix):
    return benchmark.output_root(args) / "predictions" / "published_model" / f"{prefix}.h5"


def fold_prediction_path(args, fold, prefix):
    return (
        benchmark.output_root(args)
        / "predictions"
        / "published_model"
        / f"fold_{fold}"
        / f"{prefix}.h5"
    )


def split_path(args):
    return (
        benchmark.output_root(args)
        / "split_by_allele"
        / "published_model_pred_per_snp_by_allele.joblib.gz"
    )


def ensemble_split_path(args):
    return (
        benchmark_root(args)
        / "split_by_allele"
        / "published_model_pred_per_snp_by_allele.joblib.gz"
    )


def fold_split_path(args, fold):
    return (
        benchmark.output_root(args)
        / "split_by_allele"
        / f"published_model_fold_{fold}_pred_per_snp_by_allele.joblib.gz"
    )


def score_path(args):
    return benchmark.output_root(args) / "scores" / "published_model_l2_scores.csv.gz"


def ensemble_score_path(args):
    return benchmark_root(args) / "scores" / "published_model_l2_scores.csv.gz"


def fold_score_path(args, fold):
    return (
        benchmark.output_root(args)
        / "scores"
        / f"published_model_fold_{fold}_l2_scores.csv.gz"
    )


def summary_path(args):
    return benchmark.output_root(args) / "published_clipnet_qtl_benchmark_summary.csv"


def configure_benchmark_adapter():
    benchmark.discover_checkpoints = discover_published_checkpoints
    benchmark.benchmark_root = benchmark_root
    benchmark.prediction_path = prediction_path
    benchmark.fold_prediction_path = fold_prediction_path
    benchmark.split_path = split_path
    benchmark.ensemble_split_path = ensemble_split_path
    benchmark.fold_split_path = fold_split_path
    benchmark.score_path = score_path
    benchmark.ensemble_score_path = ensemble_score_path
    benchmark.fold_score_path = fold_score_path
    benchmark.summary_path = summary_path


def resolve_fold_assignments(args):
    if args.fold_assignments is not None:
        return args.fold_assignments
    downloaded = args.model_root / "data_fold_assignments.csv"
    if downloaded.exists():
        return downloaded
    return REPO_ROOT / "clipnet_data_folds/data_fold_assignments.csv"


def add_published_reference_comparison(summary, qtl):
    expected = PUBLISHED_EXPECTED_PEARSON[qtl]
    summary = summary.copy()
    summary["published_reference_metric"] = "log_l2_pearson"
    summary["published_expected_pearson"] = expected
    summary["published_observed_pearson"] = summary["log_l2_pearson"]
    summary["published_pearson_delta"] = (
        summary["published_observed_pearson"] - expected
    )
    return summary


def annotate_summary(args):
    path = summary_path(args)
    summary = pd.read_csv(path)
    summary = add_published_reference_comparison(summary, args.qtl)
    summary.to_csv(path, index=False)
    composite = summary[summary["aggregation"] == "legacy_composite"]
    if not composite.empty:
        row = composite.iloc[-1]
        expected = row["published_expected_pearson"]
        observed = row["published_observed_pearson"]
        print(
            f"Published log-L2 Pearson reference: {expected:.3f}; "
            f"measured log-L2 Pearson: {observed:.3f}; "
            f"delta: {row['published_pearson_delta']:+.3f}; "
            f"raw-L2 Pearson diagnostic: {row['l2_pearson']:.3f}"
        )


def run_mode(args, mode):
    args.mode = mode
    if args.command in {"predict", "all"}:
        benchmark.run_predict(args)
    if args.command in {"split", "all"}:
        benchmark.run_split(args)
    if args.command in {"score", "all"}:
        benchmark.run_score(args)


def main():
    args = parse_args()
    args.fold_assignments = resolve_fold_assignments(args)
    args.best_variant = "published"
    args.checkpoint_label = "published_clipnet"
    configure_benchmark_adapter()

    requested_mode = args.mode
    if requested_mode == "composite":
        run_mode(args, "ensemble")
        run_mode(args, "folds")
    else:
        run_mode(args, requested_mode)

    if args.command in {"score", "all"}:
        args.mode = "folds" if requested_mode == "composite" else requested_mode
        annotate_summary(args)


if __name__ == "__main__":
    main()
