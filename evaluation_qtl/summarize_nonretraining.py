#!/usr/bin/env python3

"""Collect the nonretraining_cpu.sh bootstrap results into one report.

Each reported view is a (score table, statistic) pair. The headline views are
the manuscript metric (legacy composite log-L2 Pearson), the 9-model fold-0
ensemble (how CLIPNET is used), and within-fold Spearman with and without
observed coverage regressed out (calibration-robust). Shape-only L2 and the
expression-only baseline are diagnostics.

Writes, under --results_dir:
- ``summary_pairs.csv``: clipnet - ref_model differences with bootstrap CIs
  for every set, view, and checkpoint;
- ``summary_zero_l2.csv``: how often the count head clips QTL loci to zero;
- ``{qtl}_{set}_epochs.png``: per-epoch headline views with 95% CI bands;
- ``README.md``: the headline tables.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCORES = ("published_l2", "published_l2_partial", "shape_l2", "expression_only")
HEADLINE = {
    ("published_l2", "legacy_composite"): "legacy composite (manuscript)",
    ("published_l2", "fold0_ensemble"): "fold-0 ensemble",
    ("published_l2", "fold_macro_spearman"): "within-fold Spearman",
    ("published_l2_partial", "fold_macro_spearman"): "within-fold Spearman | coverage",
}
DIAGNOSTIC = {
    ("published_l2", "fold_macro"): "within-fold Pearson",
    ("shape_l2", "fold_macro"): "shape-only (within-fold Pearson)",
    ("expression_only", "fold_macro"): "expression-only baseline",
}
VIEWS = {**HEADLINE, **DIAGNOSTIC}
# Longest prefix first: set names are matched by filename prefix.
SETS = ("main", "best_model", "epochs20", "epochs100_noes", "epochs100")
COLORS = {"clipnet": "#2a78d6", "ref_model": "#eb6834"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("nonretraining_results"))
    return parser.parse_args()


def load(results_dir, kind):
    frames = []
    for qtl_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for set_name in SETS:
            for score in SCORES:
                path = qtl_dir / f"{set_name}_{score}_{kind}.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    frame.insert(0, "score", score)
                    frame.insert(0, "set", set_name)
                    frame.insert(0, "qtl", qtl_dir.name)
                    frames.append(frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        frame["view"] = [VIEWS.get((s, m)) for s, m in zip(frame.score, frame.metric)]
        frame = frame[frame.view.notna()]
    return frame


def zero_summary(results_dir):
    rows = []
    for path in sorted(results_dir.glob("*/*_zero_l2.csv")):
        stem = path.name[: -len("_zero_l2.csv")]
        set_name = next(s for s in SETS if stem.startswith(s + "_"))
        arm = stem[len(set_name) + 1:]
        zero = pd.read_csv(path)
        zero = zero[zero.checkpoint.str.match(r"^(epoch_|best_model|published_model)")
                    & (zero.fold != 0)]
        rows.append({
            "qtl": path.parent.name, "set": set_name,
            "arm": arm.replace("five_epoch_", "").replace("main_ensemble_models", "clipnet")
                      .replace("main_reference_models", "ref_model"),
            "n_checkpoint_folds": len(zero),
            "mean_frac_zero_l2": zero.frac_zero_l2.mean(),
            "frac_folds_over_half_zero": (zero.frac_zero_l2 > 0.5).mean(),
            "n_folds_all_zero": int((zero.frac_zero_l2 == 1).sum()),
        })
    return pd.DataFrame(rows)


def plot_epochs(entries, qtl, set_name, out_fp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = entries[(entries.qtl == qtl) & (entries.set == set_name)
                   & entries.checkpoint.str.startswith("epoch_")].copy()
    if data.empty:
        return False
    data["epoch"] = data.checkpoint.str[6:].astype(int)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for ax, title in zip(axes.flat, HEADLINE.values()):
        sub = data[data.view == title]
        for arm, color in COLORS.items():
            arm_data = sub[sub.label == arm].sort_values("epoch")
            if arm_data.empty:
                continue
            ax.fill_between(arm_data.epoch, arm_data.ci_low, arm_data.ci_high,
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(arm_data.epoch, arm_data.estimate, color=color, linewidth=2,
                    marker="o", markersize=4, label=arm)
        ax.set_title(title, fontsize=10, color="#333333")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#555555", labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("epoch", color="#555555")
    for ax in axes[:, 0]:
        ax.set_ylabel("r (95% SNP-bootstrap CI)", color="#555555", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{qtl}: {set_name} checkpoints", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)
    return True


def markdown_table(frame):
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    pairs = load(args.results_dir, "pairs")
    entries = load(args.results_dir, "entries")
    if pairs.empty:
        raise SystemExit(f"No bootstrap results under {args.results_dir}")
    pairs = pairs.rename(columns={"a_estimate": "clipnet", "b_estimate": "ref_model"})
    keep = ["qtl", "set", "view", "checkpoint", "n_checkpoints", "clipnet", "ref_model",
            "diff", "ci_low", "ci_high", "p_boot", "folds_a_higher"]
    pairs = pairs[keep]
    pairs.to_csv(args.results_dir / "summary_pairs.csv", index=False)
    zero = zero_summary(args.results_dir)
    zero.to_csv(args.results_dir / "summary_zero_l2.csv", index=False)

    order = {view: i for i, view in enumerate(VIEWS.values())}
    summary = pairs[~pairs.checkpoint.str.startswith("epoch_")].copy()
    summary["order"] = summary.view.map(order)
    summary = summary.sort_values(["qtl", "set", "checkpoint", "order"]).drop(
        columns=["order", "n_checkpoints"])
    headline = summary[summary.view.isin(HEADLINE.values())]
    diagnostic = summary[summary.view.isin(DIAGNOSTIC.values())]

    plots = []
    for qtl in sorted(entries.qtl.unique()):
        for set_name in ("epochs20", "epochs100", "epochs100_noes"):
            out_fp = args.results_dir / f"{qtl}_{set_name}_epochs.png"
            if plot_epochs(entries, qtl, set_name, out_fp):
                plots.append(out_fp.name)

    report = [
        "# Non-retraining clipnet vs ref_model QTL analyses",
        "",
        "Differences are clipnet - ref_model with 95% paired SNP-bootstrap CIs (SNPs "
        "resampled within folds, the same draws for both arms); they do not include "
        "training-run noise. `folds_a_higher` counts held-out folds (of 9) where "
        "clipnet's fold r is higher (within-fold views only).",
        "",
        "- **legacy composite**: the manuscript metric, log-L2 Pearson over SNPs pooled "
        "from the 9 held-out folds plus the fold-0 ensemble. Sensitive to differences in "
        "count-head calibration between fold models.",
        "- **fold-0 ensemble**: the 9-model ensemble on the fold-0 holdout (171 tiQTL / "
        "77 diQTL SNPs), as CLIPNET is used.",
        "- **within-fold Spearman** (and `| coverage`, with log observed coverage "
        "regressed out within each fold): calibration-robust.",
        "",
        "`mean_over_epochs` averages each arm's score over all shared epochs; "
        "`window_e*` scores an ensemble of the allele-mean tracks over that epoch range.",
        "",
        "## Headline comparisons",
        "",
        markdown_table(headline),
        "",
        "## Diagnostics",
        "",
        markdown_table(diagnostic),
        "",
        "## Count-head zero clipping",
        "",
        "Fraction of held-out SNPs whose predicted L2 is exactly 0 because the ReLU "
        "count head outputs 0 for every individual.",
        "",
        markdown_table(zero),
        "",
        "## Epoch traces",
        "",
        *[f"![{name}]({name})" for name in plots],
        "",
    ]
    (args.results_dir / "README.md").write_text("\n".join(report))
    with pd.option_context("display.width", 220, "display.max_rows", 400,
                           "display.float_format", "{:.3f}".format):
        print(headline.to_string(index=False))
        print(zero.to_string(index=False))


if __name__ == "__main__":
    main()
