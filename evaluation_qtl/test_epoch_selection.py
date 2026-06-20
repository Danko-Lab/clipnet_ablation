import argparse
import unittest

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


if __name__ == "__main__":
    unittest.main()
