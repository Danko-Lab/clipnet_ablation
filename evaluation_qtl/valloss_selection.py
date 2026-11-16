#!/usr/bin/env python3

"""Build a validation-loss-selected checkpoint for each checkpoint run.

For every fold model, picks the saved checkpoint with the lowest validation
loss in its training log (``fold_N.log`` or ``clipnet.log`` next to the
checkpoints; Keras rows are 0-indexed, so epoch_005 is row 4). It then writes a
benchmark directory ``<run>_valsel/folds`` whose single checkpoint ``valsel``
symlinks each fold's predictions and score table to that fold's selected epoch.
alternative_qtl_scores.py can then score it as usual, including the fold-0
ensemble (the 9 selected fold models averaged).

This mirrors how the published models were chosen (best validation loss), but
restricted to the checkpoints that were saved: every epoch for the 20-epoch
runs, every 5 epochs for the 100-epoch runs.

    python valloss_selection.py   # writes selections to nonretraining_results/valloss_selection.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ABLATION = Path("/fs/cbsubscb17/storage/projects/CLIPNET_ablation")
# bench run name -> model dir (one f<fold> subdir per fold)
RUNS = {
    "clipnet": ABLATION / "models/epochs/clipnet",
    "ref_model": ABLATION / "models/epochs/ref_model",
    "five_epoch_clipnet": ABLATION / "models/five_epoch_clipnet_clean",
    "ref100_ref_model": ABLATION / "models/ref_model_100ep",
}
EPOCH_FILE_RE = re.compile(r"^(?P<prefix>.+)_epoch_(?P<epoch>\d+)\.hdf5$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_root", type=Path, default=ABLATION / "revision_predictions")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "nonretraining_results/valloss_selection.csv")
    return parser.parse_args()


def training_log(fold_dir):
    """The val-loss log of the run whose checkpoints sit (or are linked) in fold_dir."""
    ckpt = next(p for p in sorted(fold_dir.iterdir()) if EPOCH_FILE_RE.match(p.name))
    source = ckpt.resolve()
    prefix = EPOCH_FILE_RE.match(source.name)["prefix"]
    log = source.parent / f"{prefix}.log"
    if not log.exists():
        raise FileNotFoundError(f"No training log {log} for {fold_dir}")
    return pd.read_csv(log)


def main():
    args = parse_args()
    rows = []
    for qtl in ("tiqtl", "diqtl"):
        for run, model_root in RUNS.items():
            bench = args.predictions_root / qtl / "epoch_checkpoint_benchmark" / run / "folds"
            scores = bench / "scores_published_l2"
            out = args.predictions_root / qtl / "epoch_checkpoint_benchmark" / f"{run}_valsel" / "folds"
            for fold in range(1, 10):
                saved = sorted(int(p.name.split("_")[1]) for p in scores.glob(f"epoch_*_fold_{fold}_l2_scores.csv.gz"))
                log = training_log(model_root / f"f{fold}").set_index("epoch")
                val = {e: log.loc[e - 1, "val_loss"] for e in saved if e - 1 in log.index}
                best = min(val, key=val.get)
                checkpoint = f"epoch_{best:03d}"
                rows.append({"qtl": qtl, "run": run, "fold": fold, "selected": checkpoint,
                             "val_loss": val[best], "n_saved": len(val),
                             "min_val_loss_any_epoch": log.val_loss.min(),
                             "best_epoch_any": int(log.val_loss.idxmin()) + 1})
                links = {
                    out / "predictions" / "valsel" / f"fold_{fold}":
                        bench / "predictions" / checkpoint / f"fold_{fold}",
                    out / "scores_published_l2" / f"valsel_fold_{fold}_l2_scores.csv.gz":
                        scores / f"{checkpoint}_fold_{fold}_l2_scores.csv.gz",
                }
                for link, target in links.items():
                    if not target.exists():
                        raise FileNotFoundError(target)
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if link.is_symlink():
                        link.unlink()
                    link.symlink_to(target)
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    wide = table[table.qtl == "tiqtl"].pivot(index="fold", columns="run", values="selected")
    print(wide.to_string())
    any_epoch = table[table.qtl == "tiqtl"].pivot(index="fold", columns="run", values="best_epoch_any")
    print("\nBest epoch over all epochs (not only saved):\n" + any_epoch.to_string())


if __name__ == "__main__":
    main()
