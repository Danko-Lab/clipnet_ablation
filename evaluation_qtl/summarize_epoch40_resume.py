#!/usr/bin/env python3

"""Compare epoch-45 QTL performance of clipnet vs ref_model resume replicates.

Reads the fold_macro log-L2 Pearson r of every replicate written by
epoch40_resume_benchmark.sh, draws a violin plot per arm, and tests the
difference. Replicate r shares its seed (and therefore batch order) across arms,
so a paired test is reported alongside the unpaired ones.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARMS = ("clipnet", "ref_model")
SUMMARY = "folds/epoch_qtl_benchmark_summary_published_l2.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_root", type=Path, default=Path("../predictions"))
    parser.add_argument("--qtl", choices=["tiqtl", "diqtl"], required=True)
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=45)
    parser.add_argument("--aggregation", default="fold_macro")
    parser.add_argument(
        "--baseline_stats",
        type=Path,
        default=None,
        help="qtl_benchmark_stats.csv holding the epoch-40 starting-point scores.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_replicates(args):
    rows = []
    for arm in ARMS:
        for rep in range(1, args.n_reps + 1):
            run = f"e40resume_{arm}_rep{rep:02d}"
            path = (
                args.predictions_root / args.qtl / "epoch_checkpoint_benchmark" / run / SUMMARY
            )
            if not path.exists():
                print(f"Missing {path}")
                continue
            summary = pd.read_csv(path)
            hit = summary[
                (summary.aggregation == args.aggregation) & (summary.epoch == args.epoch)
            ]
            if hit.empty:
                print(f"No {args.aggregation} row for epoch {args.epoch} in {path}")
                continue
            rows.append(
                {
                    "arm": arm,
                    "rep": rep,
                    "r": float(hit.log_l2_pearson.iloc[0]),
                    "n_usable_folds": hit.n_usable_folds.iloc[0],
                }
            )
    return pd.DataFrame(rows)


def load_baseline(path):
    if path is None or not path.exists():
        return {}
    stats_table = pd.read_csv(path)
    hit = stats_table[stats_table.label.isin(ARMS) & (stats_table.epoch == 40)]
    return dict(zip(hit.label, hit.fold_macro_r))


def compare(reps, baseline):
    by_arm = {arm: reps[reps.arm == arm].set_index("rep").r for arm in ARMS}
    clip, ref = by_arm["clipnet"], by_arm["ref_model"]
    paired = clip.index.intersection(ref.index)
    diff = clip[paired] - ref[paired]
    rows = []
    for arm, values in by_arm.items():
        rows.append(
            {
                "arm": arm,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": values.median(),
                "epoch40_start": baseline.get(arm, np.nan),
                "mean_change_from_epoch40": values.mean() - baseline.get(arm, np.nan),
            }
        )
    tests = {
        "mean_diff_clipnet_minus_ref": clip.mean() - ref.mean(),
        "welch_t_p": stats.ttest_ind(clip, ref, equal_var=False).pvalue,
        "mannwhitney_p": stats.mannwhitneyu(clip, ref, alternative="two-sided").pvalue,
        "n_paired": len(paired),
        "paired_mean_diff": diff.mean(),
        "paired_wilcoxon_p": stats.wilcoxon(diff).pvalue if len(paired) > 1 else np.nan,
        "paired_frac_clipnet_higher": (diff > 0).mean(),
    }
    return pd.DataFrame(rows), pd.DataFrame([tests])


def plot(reps, baseline, tests, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4.5))
    data = [reps[reps.arm == arm].r.to_numpy() for arm in ARMS]
    ax.violinplot(data, showmedians=True)
    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(values)), values, s=12, color="k", zorder=3)
        if ARMS[i - 1] in baseline:
            ax.scatter(i, baseline[ARMS[i - 1]], marker="D", s=50, color="tab:red", zorder=4,
                       label="epoch 40 start" if i == 1 else None)
    ax.set_xticks([1, 2], ARMS)
    ax.set_ylabel(f"log-L2 Pearson r ({args.aggregation})")
    p = tests.paired_wilcoxon_p.iloc[0]
    ax.set_title(f"{args.qtl}, epoch {args.epoch} (paired Wilcoxon p={p:.2g})", fontsize=10)
    if baseline:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / f"epoch40_resume_{args.qtl}_violin.{ext}", dpi=200)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reps = load_replicates(args)
    if reps.empty or reps.arm.nunique() < 2:
        raise SystemExit("Need replicate scores for both arms.")
    baseline = load_baseline(args.baseline_stats)
    per_arm, tests = compare(reps, baseline)
    reps.to_csv(args.output_dir / f"epoch40_resume_{args.qtl}_replicates.csv", index=False)
    per_arm.to_csv(args.output_dir / f"epoch40_resume_{args.qtl}_per_arm.csv", index=False)
    tests.to_csv(args.output_dir / f"epoch40_resume_{args.qtl}_tests.csv", index=False)
    plot(reps, baseline, tests, args)
    print(per_arm.to_string(index=False))
    print(tests.T.to_string(header=False))


if __name__ == "__main__":
    main()
