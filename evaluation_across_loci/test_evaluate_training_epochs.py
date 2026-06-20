import unittest
import sys
import types

scipy = types.ModuleType("scipy")
scipy_spatial = types.ModuleType("scipy.spatial")
scipy_spatial_distance = types.ModuleType("scipy.spatial.distance")
scipy_spatial_distance.jensenshannon = lambda *args, **kwargs: 0
scipy_stats = types.ModuleType("scipy.stats")
scipy_stats.pearsonr = lambda *args, **kwargs: (0, 1)
scipy_stats.spearmanr = lambda *args, **kwargs: (0, 1)
tensorflow = types.ModuleType("tensorflow")
tensorflow.config = types.SimpleNamespace(
    set_visible_devices=lambda *args, **kwargs: None,
    list_physical_devices=lambda *args, **kwargs: [],
    experimental=types.SimpleNamespace(
        set_memory_growth=lambda *args, **kwargs: None
    ),
)
clipnet = types.ModuleType("clipnet")
clipnet.utils = types.SimpleNamespace()

sys.modules.setdefault("scipy", scipy)
sys.modules.setdefault("scipy.spatial", scipy_spatial)
sys.modules.setdefault("scipy.spatial.distance", scipy_spatial_distance)
sys.modules.setdefault("scipy.stats", scipy_stats)
sys.modules.setdefault("tensorflow", tensorflow)
sys.modules.setdefault("clipnet", clipnet)

import evaluate_training_epochs as eval_epochs


class CrossLociEpochSelectionTest(unittest.TestCase):
    def test_default_selects_complete_epochs_only(self):
        checkpoints = {
            5: {1: "f1_e5", 2: "f2_e5"},
            10: {1: "f1_e10"},
            15: {1: "f1_e15", 2: "f2_e15"},
        }

        self.assertEqual(
            eval_epochs.select_epochs(checkpoints, [1, 2], None, False),
            [5, 15],
        )

    def test_allow_missing_keeps_partial_epochs(self):
        checkpoints = {
            5: {1: "f1_e5", 2: "f2_e5"},
            10: {1: "f1_e10"},
        }

        self.assertEqual(
            eval_epochs.select_epochs(checkpoints, [1, 2], None, True),
            [5, 10],
        )


if __name__ == "__main__":
    unittest.main()
