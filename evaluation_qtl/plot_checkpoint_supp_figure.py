#!/usr/bin/env python3

"""Supplementary figure: clipnet vs ref_model QTL performance across training checkpoints.

Reads the nonretraining_cpu.sh outputs for the 100-epoch runs without early
stopping (set ``epochs100_noes``: clipnet five_epoch runs vs the ref_model_100ep
reruns, a checkpoint every 5 epochs) under --results_dir and writes
``supp_fig_checkpoints.{pdf,png}`` there. The 20-epoch runs are left out: they
had not converged (validation loss was still falling at epochs 19-20).

Layout (columns: tiQTL, diQTL):
- row 1: legacy composite log-L2 Pearson (the manuscript metric);
- row 2: within-fold Spearman with log observed coverage regressed out;
  both with 95% paired SNP-bootstrap CIs and the published models dashed;
- row 3: mean fraction of held-out SNPs (folds 1-9) whose predicted L2 is
  exactly 0 because the ReLU count head clips every individual to 0.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARMS = {"clipnet": ("CLIPNET", "#2a78d6"), "ref_model": ("CLIPNET-ref", "#eb6834")}
METRICS = {
    ("published_l2", "legacy_composite"): "Composite log-L2 Pearson r",
    ("published_l2_partial", "fold_macro_spearman"): "Spearman ρ | coverage",
}
RUN = "epochs100_noes"
ZERO_FILES = {"clipnet": "epochs100_noes_five_epoch_clipnet_zero_l2.csv",
              "ref_model": "epochs100_noes_ref100_ref_model_zero_l2.csv"}
QTLS = {"tiqtl": "tiQTL", "diqtl": "diQTL"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path,
                        default=Path(__file__).parent / "nonretraining_results")
    return parser.parse_args()


def epoch_entries(results_dir, qtl, score, metric):
    frame = pd.read_csv(results_dir / qtl / f"{RUN}_{score}_entries.csv")
    frame = frame[(frame.metric == metric) & frame.checkpoint.str.startswith("epoch_")].copy()
    frame["epoch"] = frame.checkpoint.str[6:].astype(int)
    return frame.sort_values("epoch")


def published(results_dir, qtl, score, metric):
    frame = pd.read_csv(results_dir / qtl / f"main_{score}_entries.csv")
    frame = frame[frame.metric == metric]
    return dict(zip(frame.label, frame.estimate))


def zero_fraction(results_dir, qtl, arm):
    frame = pd.read_csv(results_dir / qtl / ZERO_FILES[arm])
    frame = frame[frame.fold != 0]
    complete = frame.groupby("checkpoint").fold.nunique() == 9
    frame = frame[frame.checkpoint.isin(complete[complete].index)]
    out = frame.groupby("checkpoint").frac_zero_l2.mean().reset_index()
    out["epoch"] = out.checkpoint.str[6:].astype(int)
    return out.sort_values("epoch")


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.tick_params(colors=MUTED, labelsize=7, length=2.5)
    ax.set_xticks([5, 25, 50, 75, 100])


def main():
    args = parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(5.6, 6.4), sharex=True)

    for col, (qtl, qname) in enumerate(QTLS.items()):
        for row, ((score, metric), label) in enumerate(METRICS.items()):
            ax = axes[row, col]
            ref_lines = published(args.results_dir, qtl, score, metric)
            data = epoch_entries(args.results_dir, qtl, score, metric)
            for arm, (name, color) in ARMS.items():
                sub = data[data.label == arm]
                ax.fill_between(sub.epoch, sub.ci_low, sub.ci_high, color=color,
                                alpha=0.16, linewidth=0)
                ax.plot(sub.epoch, sub.estimate, color=color, linewidth=1.4,
                        marker="o", markersize=2.8, label=name)
                ax.axhline(ref_lines[arm], color=color, linewidth=0.9,
                           linestyle=(0, (3, 2)), alpha=0.9)
            style(ax)
            if row == 0:
                ax.set_title(qname, fontsize=9, fontweight="bold", color=INK, pad=6)
            if col == 0:
                ax.set_ylabel(label, color=INK, fontsize=8)
        ax = axes[2, col]
        for arm, (name, color) in ARMS.items():
            sub = zero_fraction(args.results_dir, qtl, arm)
            ax.plot(sub.epoch, sub.frac_zero_l2, color=color, linewidth=1.4,
                    marker="o", markersize=2.8, label=name)
        ax.set_ylim(-0.02, 0.7)
        style(ax)
        ax.set_xlabel("Epoch", color=INK, fontsize=8)
        if col == 0:
            ax.set_ylabel("Fraction of SNPs\nwith predicted L2 = 0", color=INK, fontsize=8)

    # Same metric: share the y-range across tiQTL and diQTL.
    for row in range(2):
        lo = min(ax.get_ylim()[0] for ax in axes[row])
        hi = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(lo, hi)

    for i, ax in enumerate(axes.flat):
        ax.text(-0.16, 1.04, chr(ord("a") + i), transform=ax.transAxes,
                fontsize=9, fontweight="bold", ha="right", va="bottom", color=INK)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=0.9, linestyle=(0, (3, 2))))
    labels.append("Published model")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2, w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(args.results_dir / f"supp_fig_checkpoints.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
