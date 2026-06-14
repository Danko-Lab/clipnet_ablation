import argparse
import gzip
import io
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

import benchmark_published_clipnet as published
from published_qtl_targets import (
    load_published_experimental_l2,
    score_directory_name,
    summary_suffix,
)


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
                "aggregation": ["legacy_composite"],
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
            0.44 - published.PUBLISHED_EXPECTED_PEARSON["diqtl"][
                "legacy_composite"
            ],
        )

    def test_ensemble_uses_ensemble_reference(self):
        summary = pd.DataFrame(
            {
                "aggregation": ["ensemble"],
                "l2_pearson": [0.49],
                "log_l2_pearson": [0.50],
            }
        )

        compared = published.add_published_reference_comparison(
            summary, "tiqtl"
        )

        self.assertAlmostEqual(
            compared.loc[0, "published_expected_pearson"],
            published.PUBLISHED_EXPECTED_PEARSON["tiqtl"]["ensemble"],
        )

    def test_fold_calibrated_rows_do_not_claim_manuscript_references(self):
        summary = pd.DataFrame(
            {
                "aggregation": [
                    "fold_macro",
                    "fold_standardized_pooled",
                    "pooled_folds",
                ],
                "log_l2_pearson": [0.61, 0.59, 0.48],
            }
        )

        compared = published.add_published_reference_comparison(
            summary, "tiqtl"
        )

        self.assertTrue(
            pd.isna(
                compared.loc[
                    compared["aggregation"] == "fold_macro",
                    "published_expected_pearson",
                ]
            ).all()
        )
        self.assertAlmostEqual(
            compared.loc[
                compared["aggregation"] == "pooled_folds",
                "published_expected_pearson",
            ].iloc[0],
            published.PUBLISHED_EXPECTED_PEARSON["tiqtl"]["pooled_folds"],
        )

    def test_requires_fold_calibrated_rows_for_fold_mode(self):
        summary = pd.DataFrame(
            {"aggregation": ["fold", "pooled_folds", "legacy_composite"]}
        )

        with self.assertRaisesRegex(
            ValueError, "missing fold-calibrated aggregations"
        ):
            published.validate_fold_calibrated_summary(summary, "folds")

        published.validate_fold_calibrated_summary(summary, "ensemble")

    def test_prints_fold_calibrated_metrics(self):
        summary = pd.DataFrame(
            [
                {
                    "aggregation": "fold_macro",
                    "log_l2_pearson": 0.61,
                    "n_usable_folds": 9,
                },
                {
                    "aggregation": "fold_standardized_pooled",
                    "log_l2_pearson": 0.59,
                    "calibration_penalty": 0.11,
                },
                {
                    "aggregation": "pooled_folds",
                    "log_l2_pearson": 0.48,
                },
            ]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            published.print_fold_calibrated_summary(summary)

        self.assertIn("fold macro=0.610", output.getvalue())
        self.assertIn("calibration penalty=0.110", output.getvalue())

    def test_coordinate_header_maps_to_centered_snp(self):
        sequence_id = (
            "GSM3004689_53278_R1_cap.clip.ACTTGA.hs37d5.bwa.uniqueUMI.gz_"
            "chr10:100185757-100186756"
        )
        coordinate_to_snp = {("chr10", 100186257): ["rs10786583"]}

        self.assertEqual(
            published.benchmark.match_sequence_coordinate(
                sequence_id, coordinate_to_snp, set()
            ),
            "rs10786583",
        )

    def test_loads_published_experimental_l2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "qtl_analysis.tar.gz"
            csv_bytes = gzip.compress(b",expt,pred\nrs1,1.25,0.5\n")
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo(
                    "qtl_analysis/diqtls_ensemble_l2_scores.csv.gz"
                )
                info.size = len(csv_bytes)
                archive.addfile(info, io.BytesIO(csv_bytes))

            targets = load_published_experimental_l2(
                archive_path, "diqtl"
            )

            self.assertEqual(list(targets.index), ["rs1"])
            self.assertEqual(targets.loc["rs1"], 1.25)

    def test_archive_scores_use_separate_namespace(self):
        args = argparse.Namespace(
            experimental_l2_archive=Path("qtl_analysis.tar.gz")
        )

        self.assertEqual(score_directory_name(args), "scores_published_l2")
        self.assertEqual(summary_suffix(args), "_published_l2")

if __name__ == "__main__":
    unittest.main()
