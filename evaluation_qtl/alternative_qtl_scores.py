#!/usr/bin/env python3

"""Alternative QTL scores from existing fold-mode benchmark predictions.

Works on a fold-mode benchmark directory (``.../{run}/folds``) whose raw
predictions (``predictions/{checkpoint}/fold_{k}/{prefix}.h5``) and published-L2
score tables (``scores_published_l2/{checkpoint}_fold_{k}_l2_scores.csv.gz``)
already exist. No model is run.

``summarize`` reduces every checkpoint and fold to per-SNP homozygous-allele
means of (a) the scaled track (profile / sum x quantity, as in the benchmark)
and (b) the per-individual sum-normalized profile, cached under
``allele_means/``.

``score`` writes per-fold score tables in the benchmark format, each scored on
the same held-out SNPs as the published-L2 table:

- ``alt_scores/published_l2``: L2 of the scaled tracks (reproduces the
  benchmark; used as a consistency check) plus checkpoint-window ensembles.
- ``alt_scores/shape_l2``: L2 of the normalized profiles against the L2 of the
  sum-normalized mean experimental tracks. The count head does not enter, so
  checkpoints whose ReLU count head outputs 0 (scaled track all zero, L2 = 0)
  are still scored on their profile predictions.
- ``alt_scores/published_l2_partial``: published L2 with both log-L2 columns
  residualized on log observed coverage within each fold, so Pearson r is the
  partial correlation given expression level (observed coverage alone
  correlates with observed log L2 at r ~ 0.7).
- ``alt_scores/expression_only`` and ``alt_scores/quantity_diff``: baselines
  that predict the published L2 from the mean predicted count, or from the
  absolute count difference between homozygotes, with no profile information.

With --universe, fold 0 holds the pipeline's fold-0 ensemble: the scored SNPs
that no fold 1-9 model holds out, predicted by averaging the 9 fold models' raw
profiles and counts before scaling. It is computed from the fold-mode
predictions, so no ensemble-mode inference is needed.

A window ensemble averages the per-checkpoint allele-mean tracks over the
checkpoints in an epoch range (named ``window_e{first}-e{last}``).

The ``zero_l2`` table lists, per checkpoint and fold, the fraction of SNPs whose
predicted published L2 is exactly 0 because the count head output 0.
"""

import argparse
import logging
import re
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_epoch_checkpoints as benchmark

SCORE_RE = re.compile(r"^(?P<checkpoint>.+)_fold_(?P<fold>\d+)_l2_scores\.csv\.gz$")
EPOCH_RE = re.compile(r"^epoch_(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["summarize", "score", "all"])
    parser.add_argument("bench_dir", type=Path, help="Fold-mode benchmark directory.")
    parser.add_argument("--qtl", choices=sorted(benchmark.QTL_CONFIG), required=True)
    parser.add_argument("--qtl_data_dir", type=Path, default=None)
    parser.add_argument("--windows", default=None,
                        help="Comma-separated epoch ranges such as 1-5,6-10,11-20.")
    parser.add_argument("--checkpoints", default=None,
                        help="Comma-separated checkpoint names to process (default: all).")
    parser.add_argument("--universe", type=Path, default=None,
                        help="Ensemble-mode score table listing every scored SNP (expt, "
                             "chrom, start). With it, fold 0 is scored as the pipeline's "
                             "fold-0 ensemble: SNPs held out by no fold 1-9 model, "
                             "predicted by the 9-model average.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    args.qtl_data_dir = args.qtl_data_dir or benchmark.REPO_ROOT / "data" / args.qtl
    args.qtl_snp_bed = None
    args.qtl_table = None
    return args


def published_score_files(bench_dir):
    tasks = {}
    for path in sorted((bench_dir / "scores_published_l2").iterdir()):
        match = SCORE_RE.match(path.name)
        if match and int(match["fold"]) != 0:
            tasks[(match["checkpoint"], int(match["fold"]))] = path
    return tasks


def cache_path(bench_dir, checkpoint, fold):
    return bench_dir / "allele_means" / f"{checkpoint}_fold_{fold}.npz"


_ROW_CACHE = {}


def prefix_rows(args, pred_fp, prefix, all_snps, coordinate_to_snp):
    """Map prediction rows to SNPs; the FASTA layout is fixed per individual."""
    if prefix not in _ROW_CACHE:
        _ROW_CACHE[prefix] = benchmark.prediction_row_order(
            pred_fp, all_snps, coordinate_to_snp=coordinate_to_snp
        )
    return _ROW_CACHE[prefix]


def fold0_table(args, tasks, checkpoint):
    """Fold-0 ensemble SNPs: scored SNPs that no fold 1-9 model holds out."""
    folds = [f for c, f in tasks if c == checkpoint]
    if args.universe is None or sorted(folds) != list(range(1, 10)):
        return None
    universe = pd.read_csv(args.universe, index_col=0)
    held_out = set().union(*(pd.read_csv(tasks[(checkpoint, f)], index_col=0).index
                             for f in folds))
    return universe.loc[~universe.index.isin(held_out), ["expt", "chrom", "start"]]


def build_tasks(args, score_files):
    """(checkpoint, fold) -> (SNPs, target table, prediction dirs averaged)."""
    tasks = {}
    for (checkpoint, fold), score_fp in score_files.items():
        table = pd.read_csv(score_fp, index_col=0)
        tasks[(checkpoint, fold)] = (
            table, [args.bench_dir / "predictions" / checkpoint / f"fold_{fold}"])
    for checkpoint in sorted({c for c, _ in score_files}):
        table = fold0_table(args, score_files, checkpoint)
        if table is not None and not table.empty:
            tasks[(checkpoint, 0)] = (
                table, [args.bench_dir / "predictions" / checkpoint / f"fold_{f}"
                        for f in range(1, 10)])
    return tasks


def summarize_one(item, args, allele_matrix, coordinate_to_snp):
    """Per-SNP homozygous-allele means for one checkpoint and fold.

    With several prediction dirs (fold 0), the raw profiles and counts are
    averaged over the fold models before scaling, as in the benchmark's
    ensemble mode.
    """
    import h5py

    (checkpoint, fold), (table, pred_dirs) = item
    out_fp = cache_path(args.bench_dir, checkpoint, fold)
    if args.skip_existing and out_fp.exists():
        return
    snps = list(table.index)
    snp_idx = {snp: i for i, snp in enumerate(snps)}
    all_snps = list(allele_matrix.columns)
    missing = sum(not (d / f"{p}.h5").exists() for d in pred_dirs for p in allele_matrix.index)
    if missing:
        return f"{checkpoint} fold {fold}: {missing} prediction files missing; skipped"

    scaled = shape = quantity = None
    count = np.zeros((2, len(snps)))
    for prefix in allele_matrix.index:
        track = q = None
        for pred_dir in pred_dirs:
            pred_fp = pred_dir / f"{prefix}.h5"
            with h5py.File(pred_fp, "r") as pred:
                if pred.attrs.get("track_is_scaled", False):
                    raise ValueError(f"{pred_fp} stores scaled tracks; raw profiles are needed.")
                t = pred["track"][:].astype(np.float64)
                qq = pred["quantity"][:].astype(np.float64).reshape(len(t))
            track = t if track is None else track + t
            q = qq if q is None else q + qq
        track /= len(pred_dirs)
        q /= len(pred_dirs)
        if scaled is None:
            scaled = np.zeros((2, len(snps), track.shape[1]))
            shape = np.zeros_like(scaled)
            quantity = np.zeros((2, len(snps)))
        alleles = allele_matrix.loc[prefix]
        norm = track / (track.sum(axis=1, keepdims=True) + benchmark.PROFILE_PSEUDOCOUNT)
        for row, snp in prefix_rows(args, pred_fp, prefix, all_snps, coordinate_to_snp):
            i = snp_idx.get(snp)
            allele = alleles[snp]
            if i is None or allele not in (0, 1):
                continue
            a = int(allele)
            scaled[a, i] += norm[row] * q[row]
            shape[a, i] += norm[row]
            quantity[a, i] += q[row]
            count[a, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled /= count[..., None]
        shape /= count[..., None]
        quantity /= count
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_fp, snps=np.array(snps), scaled=scaled.astype(np.float32),
             shape=shape.astype(np.float32), quantity=quantity, count=count)


def run_summarize(args, tasks):
    allele_matrix = benchmark.load_allele_matrix(args)
    coordinate_to_snp = benchmark.load_snp_coordinate_map(args, list(allele_matrix.columns))
    work = partial(summarize_one, args=args, allele_matrix=allele_matrix,
                   coordinate_to_snp=coordinate_to_snp)
    # Fold-0 ensembles read 9x the files; start them first.
    order = sorted(tasks.items(), key=lambda item: (item[0][1] != 0, item[0]))
    with Pool(args.workers) as pool:
        for n, message in enumerate(pool.imap_unordered(work, order), 1):
            if message:
                print(message, flush=True)
            if n % 20 == 0 or n == len(tasks):
                print(f"summarized {n}/{len(tasks)}", flush=True)


def experimental_targets(args):
    """Shape L2 and mean coverage of the homozygous experimental tracks.

    Shape L2 is the L2 between the sum-normalized mean tracks of the two
    homozygotes; coverage is the mean total signal of the two homozygotes.
    """
    import joblib

    expt = joblib.load(args.qtl_data_dir / benchmark.QTL_CONFIG[args.qtl]["expt"])
    values, coverage = {}, {}
    for snp, groups in expt.items():
        if groups[0] is None or groups[1] is None:
            continue
        ref, alt = (np.asarray(g, dtype=np.float64).mean(axis=0) for g in groups[:2])
        if not (ref.size and alt.size):
            continue
        coverage[snp] = float((ref.sum() + alt.sum()) / 2)
        ref = ref / (ref.sum() + benchmark.PROFILE_PSEUDOCOUNT)
        alt = alt / (alt.sum() + benchmark.PROFILE_PSEUDOCOUNT)
        values[snp] = float(np.sqrt(np.sum((ref - alt) ** 2)))
    return pd.Series(values), pd.Series(coverage)


def l2(means):
    return np.sqrt(np.sum((means[0] - means[1]) ** 2, axis=-1))


def residualize(values, covariate):
    design = np.column_stack([np.ones_like(covariate), covariate])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def write_scores(out_fp, published, expt, pred, log_covariate=None):
    """Write a benchmark-format score table.

    With ``log_covariate``, the log-L2 columns hold residuals after regressing
    each on it within this fold, so their Pearson r is a partial correlation.
    """
    scores = pd.DataFrame({"expt": expt, "pred": pred}, index=published.index)
    scores["expt_log_l2"] = benchmark.log_l2_values(scores.expt.to_numpy())
    scores["pred_log_l2"] = benchmark.log_l2_values(scores.pred.to_numpy())
    if log_covariate is not None:
        for column in ("expt_log_l2", "pred_log_l2"):
            scores[column] = residualize(scores[column].to_numpy(), log_covariate)
    scores["chrom"] = published.chrom
    scores["start"] = published.start
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_fp)


def write_metrics(out_root, name, published, expt_targets, data, pred_published=None):
    """Write one score table per metric for a checkpoint (or window) and fold."""
    expt_shape, expt_coverage = expt_targets
    quantity = np.asarray(data["quantity"])
    pred_l2 = l2(data["scaled"]) if pred_published is None else pred_published
    log_coverage = np.log(expt_coverage.reindex(published.index).to_numpy() + 1e-3)
    write_scores(out_root / "published_l2_partial" / name, published, published.expt,
                 pred_l2, log_covariate=log_coverage)
    preds = {
        "published_l2": (published.expt, pred_l2),
        "shape_l2": (expt_shape.reindex(published.index).to_numpy(), l2(data["shape"])),
        # Baselines for the published target with no allele-specific profile
        # information: overall predicted expression, and the count change only.
        "expression_only": (published.expt, quantity.mean(axis=0)),
        "quantity_diff": (published.expt, np.abs(quantity[1] - quantity[0])),
    }
    for metric, (expt, pred) in preds.items():
        write_scores(out_root / metric / name, published, expt, pred)


def parse_windows(text, epochs):
    windows = {}
    for item in (text or "").split(","):
        if not item:
            continue
        first, last = (int(x) for x in item.split("-"))
        members = [e for e in epochs if first <= e <= last]
        if members:
            windows[f"window_e{first:03d}-e{last:03d}"] = members
    return windows


def run_score(args, tasks):
    expt_targets = experimental_targets(args)
    out_root = args.bench_dir / "alt_scores"
    zero_rows, check_rows = [], []
    cached = {}
    for (checkpoint, fold), (published, _) in sorted(tasks.items()):
        if not cache_path(args.bench_dir, checkpoint, fold).exists():
            continue
        data = np.load(cache_path(args.bench_dir, checkpoint, fold))
        if list(data["snps"]) != list(published.index):
            raise ValueError(f"SNP order changed for {checkpoint} fold {fold}.")
        cached[(checkpoint, fold)] = data
        pred_pub = l2(data["scaled"])
        if "pred" in published:
            check_rows.append({"checkpoint": checkpoint, "fold": fold,
                               "max_abs_diff_vs_benchmark": float(np.nanmax(np.abs(pred_pub - published.pred)))})
        zero_rows.append({"checkpoint": checkpoint, "fold": fold, "n_snps": len(published),
                          "frac_zero_l2": float((pred_pub == 0).mean()),
                          "frac_zero_quantity": float((data["quantity"] == 0).any(axis=0).mean())})
        write_metrics(out_root, f"{checkpoint}_fold_{fold}_l2_scores.csv.gz", published,
                      expt_targets, data, pred_pub)

    epochs = sorted({int(m[1]) for c, _ in tasks if (m := EPOCH_RE.match(c))})
    folds = sorted({f for _, f in tasks})
    for name, members in parse_windows(args.windows, epochs).items():
        for fold in folds:
            keys = [(f"epoch_{e:03d}", fold) for e in members]
            if any(k not in cached for k in keys):
                print(f"{name}: fold {fold} lacks some epochs; skipping.")
                continue
            published = tasks[keys[0]][0]
            mean = {field: np.mean([cached[k][field] for k in keys], axis=0)
                    for field in ("scaled", "shape", "quantity")}
            write_metrics(out_root, f"{name}_fold_{fold}_l2_scores.csv.gz", published,
                          expt_targets, mean)

    pd.DataFrame(zero_rows).to_csv(out_root / "zero_l2.csv", index=False)
    check = pd.DataFrame(check_rows)
    check.to_csv(out_root / "published_l2_consistency.csv", index=False)
    print(f"Largest |recomputed - benchmark| published L2: "
          f"{check.max_abs_diff_vs_benchmark.max():.2e}")


def main():
    # The benchmark logs every unique-remainder SNP mapping once per file; the
    # mapping itself is identical to the benchmark's.
    logging.getLogger().setLevel(logging.ERROR)
    args = parse_args()
    score_files = published_score_files(args.bench_dir)
    if args.checkpoints is not None:
        keep = set(args.checkpoints.split(","))
        score_files = {key: fp for key, fp in score_files.items() if key[0] in keep}
    tasks = build_tasks(args, score_files)
    print(f"{len(tasks)} checkpoint-fold score tables in {args.bench_dir}")
    if args.command in ("summarize", "all"):
        run_summarize(args, tasks)
    if args.command in ("score", "all"):
        run_score(args, tasks)


if __name__ == "__main__":
    main()
