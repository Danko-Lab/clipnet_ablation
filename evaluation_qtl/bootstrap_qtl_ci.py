#!/usr/bin/env python3

"""Bootstrap confidence intervals for fold-mode QTL benchmark scores.

Reads the per-fold per-SNP score tables written by the benchmark scripts
(``{checkpoint}_fold_{k}_l2_scores.csv.gz``), resamples SNPs within each held-out
fold, and recomputes:

- ``fold_macro``: Fisher-averaged per-fold log-L2 Pearson r (the primary
  statistic, identical to fold_calibrated_scoring.py).
- ``fold_macro_spearman``: the same with per-fold Spearman rho.
- ``fold_standardized_pooled``: Pearson r after z-scoring within each fold,
  which equals the (n_k - 1)-weighted mean of per-fold r.
- ``pooled_folds``: Pearson r on the raw log L2 pooled over folds 1-9.
- ``legacy_composite`` (the manuscript metric) and ``fold0_ensemble``: as in
  the benchmark, when fold-0 ensemble scores (``*_fold_0_*``) are present.

A fold where an entry predicts a constant (r undefined) is scored as r = 0;
the benchmark summaries instead drop that fold from fold_macro.

Every entry in a run is resampled with the same SNP draws, so differences
between entries are paired. The intervals describe SNP sampling only: they do
not include run-to-run or checkpoint-to-checkpoint training noise.

Entries are given as LABEL=SCORE_DIR (every checkpoint found in the directory)
or LABEL=SCORE_DIR@CHECKPOINT (one checkpoint). Labels that expand to several
checkpoints are compared checkpoint-by-checkpoint, and also on their mean over
the shared single-epoch checkpoints (``epoch_*``). Single-checkpoint entries are compared with --pair.

    python bootstrap_qtl_ci.py \\
        --entry clipnet=.../clipnet/folds/scores_published_l2 \\
        --entry ref_model=.../ref_model/folds/scores_published_l2 \\
        --output_dir tiqtl_bootstrap
"""

import argparse
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCORE_RE = re.compile(r"^(?P<checkpoint>.+)_fold_(?P<fold>\d+)_l2_scores\.csv\.gz$")
METRICS = ("legacy_composite", "fold0_ensemble", "pooled_folds", "fold_macro",
           "fold_macro_spearman", "fold_standardized_pooled")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--entry", action="append", required=True,
                        help="LABEL=SCORE_DIR or LABEL=SCORE_DIR@CHECKPOINT.")
    parser.add_argument("--pair", action="append", default=[],
                        help="A:B labels to compare (default: every pair of labels).")
    parser.add_argument("--checkpoints", default=None,
                        help="Comma-separated checkpoints to keep for multi-checkpoint labels.")
    parser.add_argument("--exclude_folds", default=None,
                        help="Comma-separated folds to leave out, for sensitivity "
                             "analyses next to the full 9-fold result.")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prefix", default="bootstrap")
    return parser.parse_args()


def discover(score_dir):
    found = defaultdict(dict)
    for path in sorted(Path(score_dir).iterdir()):
        match = SCORE_RE.match(path.name)
        if match:
            found[match["checkpoint"]][int(match["fold"])] = path
    if not found:
        raise FileNotFoundError(f"No *_fold_N_l2_scores.csv.gz files in {score_dir}")
    return found


def load_entries(specs, keep_checkpoints):
    """Return {label: {checkpoint: {fold: DataFrame}}}."""
    entries = {}
    for spec in specs:
        label, _, location = spec.partition("=")
        score_dir, _, checkpoint = location.partition("@")
        found = discover(score_dir)
        if checkpoint:
            if checkpoint not in found:
                raise KeyError(f"{checkpoint} not in {score_dir}: {sorted(found)}")
            found = {checkpoint: found[checkpoint]}
        elif keep_checkpoints is not None:
            found = {c: f for c, f in found.items() if c in keep_checkpoints}
        all_folds = set().union(*(f.keys() for f in found.values())) - {0}
        partial = sorted(c for c, f in found.items() if set(f) - {0} != all_folds)
        if partial:
            print(f"{label}: skipping checkpoints missing folds: {partial}")
            found = {c: f for c, f in found.items() if c not in partial}
        entries[label] = {
            c: {
                fold: pd.read_csv(path, index_col=0)
                for fold, path in folds.items()
            }
            for c, folds in found.items()
        }
    return entries


def common_snps(entries):
    """SNP IDs per fold present in every entry, so all draws are paired."""
    per_fold = defaultdict(list)
    for checkpoints in entries.values():
        for folds in checkpoints.values():
            for fold, scores in folds.items():
                usable = scores[np.isfinite(scores.expt_log_l2) & np.isfinite(scores.pred_log_l2)]
                per_fold[fold].append(set(usable.index))
    n_sets = sum(len(c) for c in entries.values())
    common = {}
    for fold, sets in per_fold.items():
        if len(sets) != n_sets:
            print(f"Fold {fold} is missing from some entries; dropping it.")
            continue
        common[fold] = sorted(set.intersection(*sets))
    return dict(sorted(common.items()))


def row_pearson(x, y):
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    return (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1))


def fisher_mean(r, axis):
    return np.tanh(np.mean(np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12)), axis=axis))


def fold_statistics(x, y, idx):
    """Pearson and Spearman r for the observed data and each bootstrap draw."""
    xb, yb = x[idx], y[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        point = (row_pearson(x, y), row_pearson(stats.rankdata(x), stats.rankdata(y)))
        boot = (
            row_pearson(xb, yb),
            row_pearson(stats.rankdata(xb, axis=1), stats.rankdata(yb, axis=1)),
        )
    # A constant prediction (e.g. the count head outputs 0 everywhere) has no
    # correlation; score it as r = 0 rather than dropping the fold.
    return tuple(np.nan_to_num(v, nan=0.0) for v in point), tuple(
        np.nan_to_num(v, nan=0.0) for v in boot)


def pooled_r(x, y, idx):
    """Pearson r on SNPs pooled across folds, observed and per draw."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.nan_to_num(row_pearson(x, y), nan=0.0),
                np.nan_to_num(row_pearson(x[idx], y[idx]), nan=0.0))


def bootstrap(entries, snps, n_boot, seed):
    """Return point estimates, bootstrap draws, and per-fold r for every entry.

    Within-fold statistics use folds 1-9. pooled_folds pools the raw log L2 of
    folds 1-9 (the benchmark's pooled_folds); legacy_composite also pools the
    fold-0 ensemble SNPs (the manuscript metric); fold0_ensemble scores those
    alone. The pooled statistics need fold 0 (``{checkpoint}_fold_0_*``) for
    the last two and are NaN without it.
    """
    rng = np.random.default_rng(seed)
    draws = {fold: rng.integers(0, len(ids), size=(n_boot, len(ids))) for fold, ids in snps.items()}
    held_out = [f for f in snps if f != 0]
    weights = np.array([len(snps[f]) - 1 for f in held_out], dtype=float)
    offsets = np.cumsum([0] + [len(snps[f]) for f in snps])[:-1]
    offset = dict(zip(snps, offsets))
    pooled_idx = np.hstack([draws[f] + offset[f] for f in held_out])
    composite_idx = np.hstack([draws[f] + offset[f] for f in snps])
    point, boot, fold_rows = {}, {}, []
    for label, checkpoints in entries.items():
        for checkpoint, folds in checkpoints.items():
            r_point, rho_point, r_boot, rho_boot = [], [], [], []
            xs, ys = [], []
            for fold, ids in snps.items():
                scores = folds[fold].loc[ids]
                x = scores.expt_log_l2.to_numpy(float)
                y = scores.pred_log_l2.to_numpy(float)
                xs.append(x)
                ys.append(y)
                (r, rho), (rb, rhob) = fold_statistics(x, y, draws[fold])
                fold_rows.append({"label": label, "checkpoint": checkpoint, "fold": fold,
                                  "n_snps": len(ids), "pearson": r, "spearman": rho})
                if fold == 0:
                    fold0 = (r, rb)
                    continue
                r_point.append(r)
                rho_point.append(rho)
                r_boot.append(rb)
                rho_boot.append(rhob)
            x_all, y_all = np.concatenate(xs), np.concatenate(ys)
            r_point, rho_point = np.array(r_point), np.array(rho_point)
            r_boot, rho_boot = np.stack(r_boot), np.stack(rho_boot)
            # pooled_idx indexes x_all (all folds, in snps order) but draws only
            # from folds 1-9; the observed pooled r uses the same SNPs.
            in_held = np.hstack([np.arange(offset[f], offset[f] + len(snps[f])) for f in held_out])
            pooled_point = pooled_r(x_all[in_held], y_all[in_held], np.arange(len(in_held))[None])[0]
            pooled_boot = pooled_r(x_all, y_all, pooled_idx)[1]
            if 0 in snps:
                composite_point, composite_boot = pooled_r(x_all, y_all, composite_idx)
                fold0_point, fold0_boot = fold0
            else:
                composite_point = fold0_point = np.nan
                composite_boot = fold0_boot = np.full(n_boot, np.nan)
            key = (label, checkpoint)
            point[key] = {
                "legacy_composite": composite_point,
                "fold0_ensemble": fold0_point,
                "pooled_folds": pooled_point,
                "fold_macro": fisher_mean(r_point, 0),
                "fold_macro_spearman": fisher_mean(rho_point, 0),
                "fold_standardized_pooled": np.average(r_point, weights=weights),
                "fold_r": r_point,
            }
            boot[key] = {
                "legacy_composite": composite_boot,
                "fold0_ensemble": fold0_boot,
                "pooled_folds": pooled_boot,
                "fold_macro": fisher_mean(r_boot, 0),
                "fold_macro_spearman": fisher_mean(rho_boot, 0),
                "fold_standardized_pooled": weights @ r_boot / weights.sum(),
            }
    return point, boot, pd.DataFrame(fold_rows)


def interval(values, ci):
    alpha = (1 - ci) / 2
    return np.quantile(values, alpha), np.quantile(values, 1 - alpha)


def two_sided_p(diff_boot):
    """Percentile-bootstrap p-value for a zero difference."""
    tail = min(np.mean(diff_boot <= 0), np.mean(diff_boot >= 0))
    return min(1.0, 2 * tail + 1 / (len(diff_boot) + 1))


def entry_table(point, boot, ci):
    rows = []
    for key in point:
        for metric in METRICS:
            if not np.isfinite(point[key][metric]):
                continue
            low, high = interval(boot[key][metric], ci)
            rows.append({"label": key[0], "checkpoint": key[1], "metric": metric,
                         "estimate": point[key][metric], "boot_sd": boot[key][metric].std(ddof=1),
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def compare(a_keys, b_keys, point, boot, ci, checkpoint_label):
    """Difference A - B averaged over matched entries (one entry each = a plain pair)."""
    rows = []
    for metric in METRICS:
        a_point = np.mean([point[k][metric] for k in a_keys])
        b_point = np.mean([point[k][metric] for k in b_keys])
        if not (np.isfinite(a_point) and np.isfinite(b_point)):
            continue
        diff = (np.mean([boot[k][metric] for k in a_keys], axis=0)
                - np.mean([boot[k][metric] for k in b_keys], axis=0))
        low, high = interval(diff, ci)
        row = {"a": a_keys[0][0], "b": b_keys[0][0], "checkpoint": checkpoint_label,
               "n_checkpoints": len(a_keys), "metric": metric, "a_estimate": a_point,
               "b_estimate": b_point, "diff": a_point - b_point, "ci_low": low,
               "ci_high": high, "p_boot": two_sided_p(diff)}
        if len(a_keys) == 1:
            fold_diff = point[a_keys[0]]["fold_r"] - point[b_keys[0]]["fold_r"]
            row["folds_a_higher"] = int((fold_diff > 0).sum())
            row["n_folds"] = len(fold_diff)
        rows.append(row)
    return rows


def pair_table(entries, point, boot, pairs, ci):
    rows = []
    for a, b in pairs:
        a_ckpts, b_ckpts = list(entries[a]), list(entries[b])
        if len(a_ckpts) == 1 or len(b_ckpts) == 1:
            if len(a_ckpts) != len(b_ckpts):
                raise ValueError(f"Cannot pair {a} ({len(a_ckpts)} checkpoints) with "
                                 f"{b} ({len(b_ckpts)}); use @CHECKPOINT on both.")
            label = a_ckpts[0] if a_ckpts == b_ckpts else f"{a_ckpts[0]} vs {b_ckpts[0]}"
            rows += compare([(a, a_ckpts[0])], [(b, b_ckpts[0])], point, boot, ci, label)
            continue
        shared = [c for c in a_ckpts if c in entries[b]]
        if not shared:
            print(f"No shared checkpoints between {a} and {b}; skipping.")
            continue
        for checkpoint in shared:
            rows += compare([(a, checkpoint)], [(b, checkpoint)], point, boot, ci, checkpoint)
        epochs = [c for c in shared if c.startswith("epoch_")]
        if len(epochs) > 1:
            rows += compare([(a, c) for c in epochs], [(b, c) for c in epochs], point,
                            boot, ci, "mean_over_epochs")
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    keep = None if args.checkpoints is None else set(args.checkpoints.split(","))
    entries = load_entries(args.entry, keep)
    snps = common_snps(entries)
    if args.exclude_folds:
        excluded = {int(f) for f in args.exclude_folds.split(",")}
        snps = {fold: ids for fold, ids in snps.items() if fold not in excluded}
    print("SNPs per fold:", {fold: len(ids) for fold, ids in snps.items()})
    pairs = ([tuple(p.split(":")) for p in args.pair] if args.pair
             else list(itertools.combinations(entries, 2)))

    point, boot, folds = bootstrap(entries, snps, args.n_boot, args.seed)
    per_entry = entry_table(point, boot, args.ci)
    per_pair = pair_table(entries, point, boot, pairs, args.ci)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_entry.to_csv(args.output_dir / f"{args.prefix}_entries.csv", index=False)
    per_pair.to_csv(args.output_dir / f"{args.prefix}_pairs.csv", index=False)
    folds.to_csv(args.output_dir / f"{args.prefix}_fold_r.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 500,
                           "display.float_format", "{:.3f}".format):
        macro = per_entry[per_entry.metric == "fold_macro"]
        print(macro.drop(columns="metric").to_string(index=False))
        if not per_pair.empty:
            print(per_pair.to_string(index=False))


if __name__ == "__main__":
    main()
