#!/usr/bin/env python3

"""Corrected composite QTL scatter plots and Zenodo fold tables.

The published per-fold CLIPNET QTL table (``*_individual_folds_l2_scores``)
scored held-out fold k with the cumulative average of fold models 1..k. This
rebuilds the composite (folds 1-9 held-out SNPs + fold-0 ensemble remainder)
from the correctly scored per-fold tables of the main-model rerun, and:

- plots observed vs predicted log10(L2 + 1e-3) for CLIPNET and CLIPNET-ref in
  the style of plot_reference_{ti,di}qtl.ipynb (KDE-coloured, viridis, y = x);
- writes corrected ``{ti,di}qtls_individual_folds_l2_scores.csv.gz`` in the
  Zenodo format (SNP index, expt, pred) for replacing the archive tables.

    python plot_corrected_qtl_scatter.py      # from evaluation_qtl/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, spearmanr

ROOT = Path("/fs/cbsubscb17/storage/projects/CLIPNET_ablation/revision_predictions")
MODELS = {"clipnet": ("CLIPNET", "main_ensemble_models"),
          "clipnet_ref": ("CLIPNET-ref", "main_reference_models")}
QTLS = {"tiqtl": "tiQTL", "diqtl": "diQTL"}
PSEUDO = 1e-3
LIM = (-3.1, 2.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_root", type=Path, default=ROOT)
    parser.add_argument("--out_dir", type=Path,
                        default=Path(__file__).parent / "nonretraining_results/corrected_qtl")
    return parser.parse_args()


def composite(root, qtl, run):
    d = root / qtl / "published_clipnet_benchmark" / run / "folds" / "alt_scores" / "published_l2"
    tables = [pd.read_csv(d / f"published_model_fold_{k}_l2_scores.csv.gz", index_col=0)
              .assign(fold=k) for k in range(10)]
    return pd.concat(tables)


def scatter(ax, table, title):
    x = np.log10(table.expt.to_numpy() + PSEUDO)
    y = np.log10(table.pred.to_numpy() + PSEUDO)
    density = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    order = np.argsort(density)
    ax.scatter(x[order], y[order], c=density[order], cmap="viridis", s=6, linewidths=0)
    ax.axline((0, 0), slope=1, color="black", lw=1, ls="--")
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed QTL effect, log10(L2)")
    ax.set_ylabel("Predicted QTL effect, log10(L2)")
    r, rho = pearsonr(x, y)[0], spearmanr(x, y)[0]
    ax.set_title(title, fontsize=9)
    ax.text(0.04, 0.96, f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\nn = {len(x):,}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.grid(color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    return r, rho


def main():
    args = parse_args()
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    rows = []
    for i, (qtl, qname) in enumerate(QTLS.items()):
        for j, (key, (mname, run)) in enumerate(MODELS.items()):
            table = composite(args.predictions_root, qtl, run)
            r, rho = scatter(axes[i, j], table, f"{qname}: {mname}")
            rows.append({"qtl": qtl, "model": mname, "n_snps": len(table),
                         "log_l2_pearson": r, "log_l2_spearman": rho})
            single, ax = plt.subplots(figsize=(4, 4))
            scatter(ax, table, f"{qname}: {mname}")
            single.tight_layout()
            single.savefig(args.out_dir / f"{qtl}_{key}_composite_l2.pdf")
            plt.close(single)
            if key == "clipnet":
                folds = table[table.fold != 0][["expt", "pred"]]
                folds.to_csv(args.out_dir / f"{qtl}s_individual_folds_l2_scores.csv.gz")
    for ax, letter in zip(axes.flat, "abcd"):
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"corrected_composite_l2.{ext}", dpi=300)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "corrected_composite_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
