import argparse
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import benchmark_published_clipnet as published


class PublishedClipnetBenchmarkTest(unittest.TestCase):
    def test_discovers_flat_zenodo_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir)
            for fold in range(1, 10):
                (model_root / f"fold_{fold}.h5").touch()
            (model_root / "data_fold_assignments.csv").touch()

            checkpoints = published.discover_published_checkpoints(
                argparse.Namespace(model_root=model_root)
            )

            self.assertEqual(list(checkpoints), list(range(1, 10)))
            self.assertEqual(checkpoints[4].name, "fold_4.h5")

    def test_rejects_duplicate_extensions_for_same_fold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir)
            (model_root / "fold_1.h5").touch()
            (model_root / "fold_1.hdf5").touch()

            with self.assertRaisesRegex(ValueError, "Multiple published checkpoints"):
                published.discover_published_checkpoints(
                    argparse.Namespace(model_root=model_root)
                )

    def test_uses_separate_published_output_namespace(self):
        args = argparse.Namespace(
            predictions_root=Path("/tmp/predictions"),
            qtl="tiqtl",
            run_name="published",
            mode="ensemble",
        )

        self.assertEqual(
            published.benchmark_root(args),
            Path(
                "/tmp/predictions/tiqtl/"
                "published_clipnet_benchmark/published"
            ),
        )

    def test_prefers_downloaded_fold_assignments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_root = Path(tmpdir)
            assignment_path = model_root / "data_fold_assignments.csv"
            assignment_path.touch()
            args = argparse.Namespace(
                model_root=model_root,
                fold_assignments=None,
            )

            self.assertEqual(
                published.resolve_fold_assignments(args),
                assignment_path,
            )

    def test_published_delta_uses_log_l2_pearson(self):
        summary = pd.DataFrame(
            {
                "l2_pearson": [0.53],
                "log_l2_pearson": [0.44],
            }
        )

        compared = published.add_published_reference_comparison(
            summary, "diqtl"
        )

        self.assertEqual(
            compared.loc[0, "published_reference_metric"],
            "log_l2_pearson",
        )
        self.assertAlmostEqual(
            compared.loc[0, "published_observed_pearson"],
            0.44,
        )
        self.assertAlmostEqual(
            compared.loc[0, "published_pearson_delta"],
            0.44 - 0.542,
        )


if __name__ == "__main__":
    unittest.main()
