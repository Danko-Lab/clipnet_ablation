#!/usr/bin/env python3

"""Compare a local published-CLIPNET benchmark with archived paper scores."""

import argparse
import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE_NAMES = {
    "diqtl": {
        "ensemble": "qtl_analysis/diqtls_ensemble_l2_scores.csv.gz",
        "folds": "qtl_analysis/diqtls_individual_folds_l2_scores.csv.gz",
    },
    "tiqtl": {
        "ensemble": "qtl_analysis/tiqtls_ensemble_l2_scores.csv.gz",
        "folds": "qtl_analysis/tiqtls_individual_folds_l2_scores.csv.gz",
    },
}
LOCAL_ENSEMBLE_SCORE = "published_model_l2_scores.csv.gz"
LOCAL_FOLD_GLOB = "published_model_fold_*_l2_scores.csv.gz"
LOG_PSEUDOCOUNT = 1e-3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtl", choices=sorted(ARCHIVE_NAMES), required=True)
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        required=True,
        help=(
            "Published benchmark run root, for example "
            "predictions/diqtl/published_clipnet_benchmark/zenodo_10408623."
        ),
    )
    parser.add_argument(
        "--published_archive",
        type=Path,
        required=True,
        help="Official qtl_analysis.tar.gz from Zenodo record 10597358.",
    )
    return parser.parse_args()


def load_score(path):
    score = pd.read_csv(path, index_col=0)
    required = {"expt", "pred"}
    missing = required - set(score.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if score.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate SNP labels.")
    return score.loc[:, ["expt", "pred"]]


def composite_scores(ensemble, folds):
    if folds.index.has_duplicates:
        duplicates = folds.index[folds.index.duplicated()].unique()
        raise ValueError(
            "Fold score files contain duplicate SNPs: "
            f"{list(duplicates[:5])} ({len(duplicates)} total)."
        )
    remainder = ensemble.loc[~ensemble.index.isin(folds.index)]
    return pd.concat([folds, remainder])


def load_local_composite(root):
    ensemble_path = root / "scores" / LOCAL_ENSEMBLE_SCORE
    fold_paths = sorted((root / "folds" / "scores").glob(LOCAL_FOLD_GLOB))
    if not ensemble_path.exists():
        raise FileNotFoundError(ensemble_path)
    if not fold_paths:
        raise FileNotFoundError(
            f"No {LOCAL_FOLD_GLOB} files found under {root / 'folds/scores'}."
        )
    ensemble = load_score(ensemble_path)
    folds = pd.concat([load_score(path) for path in fold_paths])
    return composite_scores(ensemble, folds), ensemble, folds


def read_archived_score(archive, member_name):
    member = archive.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(
            f"{member_name} is not present in the published archive."
        )
    contents = gzip.decompress(member.read())
    return pd.read_csv(io.BytesIO(contents), index_col=0).loc[:, ["expt", "pred"]]


def load_published_composite(path, qtl):
    names = ARCHIVE_NAMES[qtl]
    with tarfile.open(path, "r:gz") as archive:
        ensemble = read_archived_score(archive, names["ensemble"])
        folds = read_archived_score(archive, names["folds"])
    return composite_scores(ensemble, folds), ensemble, folds


def log_l2(values):
    return np.log(np.asarray(values, dtype=np.float64) + LOG_PSEUDOCOUNT)


def correlation(left, right):
    return pd.Series(left).corr(pd.Series(right), method="pearson")


def print_comparison(local, published):
    shared = local.index.intersection(published.index, sort=False)
    print(f"local composite SNPs: {len(local):,}")
    print(f"published composite SNPs: {len(published):,}")
    print(f"shared SNPs: {len(shared):,}")
    print(f"local-only SNPs: {len(local.index.difference(published.index)):,}")
    print(f"published-only SNPs: {len(published.index.difference(local.index)):,}")
    if len(shared) < 2:
        raise ValueError("Fewer than two shared SNPs; scores cannot be compared.")

    local = local.loc[shared]
    published = published.loc[shared]
    print("\nPer-SNP agreement with published archive")
    for column, label in (("expt", "observed PRO-cap"), ("pred", "model prediction")):
        raw = correlation(local[column].to_numpy(), published[column].to_numpy())
        logged = correlation(log_l2(local[column]), log_l2(published[column]))
        print(f"  {label}: raw r={raw:.6f}; log-L2 r={logged:.6f}")

    print("\nBenchmark correlations on shared SNPs")
    local_corr = correlation(log_l2(local["expt"]), log_l2(local["pred"]))
    published_corr = correlation(
        log_l2(published["expt"]), log_l2(published["pred"])
    )
    print(f"  local log-L2 Pearson: {local_corr:.6f}")
    print(f"  published log-L2 Pearson: {published_corr:.6f}")
    print(f"  delta: {local_corr - published_corr:+.6f}")


def main():
    args = parse_args()
    local, local_ensemble, local_folds = load_local_composite(args.benchmark_root)
    published, published_ensemble, published_folds = load_published_composite(
        args.published_archive, args.qtl
    )
    print(
        f"Local components: {len(local_folds):,} fold SNPs + "
        f"{len(local) - len(local_folds):,} ensemble remainder"
    )
    print(
        f"Published components: {len(published_folds):,} fold SNPs + "
        f"{len(published) - len(published_folds):,} ensemble remainder"
    )
    print_comparison(local, published)


if __name__ == "__main__":
    main()
