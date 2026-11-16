#!/usr/bin/env python3

"""Bootstrap the difference between each arm's best checkpoint.

For a checkpoint run (every epoch_* checkpoint of clipnet and ref_model), the
statistic is max_c(clipnet metric) - max_c(ref_model metric). Each bootstrap
draw resamples SNPs within folds (the same draws for every checkpoint of both
arms, as in bootstrap_qtl_ci.py) and re-selects each arm's best checkpoint on
that draw, so checkpoint selection is part of the interval. Selecting the best
checkpoint on the evaluation SNPs still favours the more volatile arm; the
interval does not remove that bias, it only reflects SNP sampling around it.

    python max_checkpoint_bootstrap.py --output nonretraining_results/max_checkpoint_bootstrap.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_qtl_ci import bootstrap, common_snps, interval, load_entries, two_sided_p

ROOT = Path("/fs/cbsubscb17/storage/projects/CLIPNET_ablation/revision_predictions")
# set label -> (clipnet bench dir, ref_model bench dir)
SETS = {
    "epochs20": ("epoch_checkpoint_benchmark/clipnet/folds",
                 "epoch_checkpoint_benchmark/ref_model/folds"),
    "epochs100_noes": ("epoch_checkpoint_benchmark/five_epoch_clipnet/folds",
                       "epoch_checkpoint_benchmark/ref100_ref_model/folds"),
}
VIEWS = {
    ("published_l2", "legacy_composite"): "legacy composite (manuscript)",
    ("published_l2", "fold0_ensemble"): "fold-0 ensemble",
    ("published_l2", "fold_macro_spearman"): "within-fold Spearman",
    ("published_l2_partial", "fold_macro_spearman"): "within-fold Spearman | coverage",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_root", type=Path, default=ROOT)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for qtl in ("tiqtl", "diqtl"):
        for set_name, (clip_dir, ref_dir) in SETS.items():
            for score in sorted({s for s, _ in VIEWS}):
                specs = [f"{arm}={args.predictions_root / qtl / d / 'alt_scores' / score}"
                         for arm, d in (("clipnet", clip_dir), ("ref_model", ref_dir))]
                entries = load_entries(specs, None)
                entries = {arm: {c: f for c, f in ckpts.items() if c.startswith("epoch_")}
                           for arm, ckpts in entries.items()}
                shared = sorted(set(entries["clipnet"]) & set(entries["ref_model"]))
                entries = {arm: {c: ckpts[c] for c in shared} for arm, ckpts in entries.items()}
                snps = common_snps(entries)
                point, boot, _ = bootstrap(entries, snps, args.n_boot, args.seed)
                for (view_score, metric), view in VIEWS.items():
                    if view_score != score:
                        continue
                    est, draws, best = {}, {}, {}
                    for arm in ("clipnet", "ref_model"):
                        values = np.array([point[(arm, c)][metric] for c in shared])
                        est[arm] = values.max()
                        best[arm] = shared[int(values.argmax())]
                        draws[arm] = np.stack([boot[(arm, c)][metric] for c in shared]).max(axis=0)
                    diff = draws["clipnet"] - draws["ref_model"]
                    low, high = interval(diff, args.ci)
                    rows.append({
                        "qtl": qtl, "set": set_name, "view": view, "n_checkpoints": len(shared),
                        "clipnet_max": est["clipnet"], "clipnet_best": best["clipnet"],
                        "ref_model_max": est["ref_model"], "ref_model_best": best["ref_model"],
                        "diff": est["clipnet"] - est["ref_model"], "ci_low": low,
                        "ci_high": high, "p_boot": two_sided_p(diff),
                    })
                    print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
