import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    import benchmark_epoch_checkpoints as qtl_epochs
except ImportError:
    from evaluation_qtl import benchmark_epoch_checkpoints as qtl_epochs


class QtlEpochSelectionTest(unittest.TestCase):
    def test_default_selects_complete_epochs_only(self):
        args = argparse.Namespace(epochs=None, allow_missing_folds=False)
        checkpoints = {
            5: {1: "f1_e5", 2: "f2_e5"},
            10: {1: "f1_e10"},
            15: {1: "f1_e15", 2: "f2_e15"},
        }

        self.assertEqual(
            qtl_epochs.selected_epochs(args, checkpoints, [1, 2]),
            [5, 15],
        )

    def test_allow_missing_keeps_partial_epochs(self):
        args = argparse.Namespace(epochs=None, allow_missing_folds=True)
        checkpoints = {
            5: {1: "f1_e5", 2: "f2_e5"},
            10: {1: "f1_e10"},
        }

        self.assertEqual(
            qtl_epochs.selected_epochs(args, checkpoints, [1, 2]),
            [5, 10],
        )

    def test_requested_epochs_still_require_checkpoint_epoch(self):
        args = argparse.Namespace(epochs=[5, 20], allow_missing_folds=True)
        checkpoints = {5: {1: "f1_e5"}}

        with self.assertRaisesRegex(FileNotFoundError, "No checkpoints found"):
            qtl_epochs.selected_epochs(args, checkpoints, [1])

    def test_best_fold_epoch_scores_selects_best_fold_specific_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = argparse.Namespace(
                mode="folds",
                predictions_root=root,
                qtl="tiqtl",
                model_root=Path("model"),
                run_name="run",
                experimental_l2_archive=None,
            )
            score_root = (
                root
                / "tiqtl"
                / "epoch_checkpoint_benchmark"
                / "run"
                / "folds"
                / "scores"
            )
            score_root.mkdir(parents=True)
            pd.DataFrame(
                {"expt": [1.0, 2.0, 3.0], "pred": [3.0, 2.0, 1.0]},
                index=["a", "b", "c"],
            ).to_csv(score_root / "epoch_005_fold_1_l2_scores.csv.gz")
            pd.DataFrame(
                {"expt": [1.0, 2.0, 3.0], "pred": [1.0, 2.0, 3.0]},
                index=["a", "b", "c"],
            ).to_csv(score_root / "epoch_010_fold_1_l2_scores.csv.gz")

            scores, rows = qtl_epochs.best_fold_epoch_scores(
                args, [5, 10], [1], ["a", "b", "c"]
            )

            self.assertEqual(list(scores), [1])
            self.assertEqual(scores[1]["selected_epoch"].iloc[0], 10)
            self.assertEqual(rows[0]["selected_epoch"], 10)
            self.assertEqual(rows[0]["aggregation"], "best_per_fold")


if __name__ == "__main__":
    unittest.main()
