import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from summarize_qtl_benchmarks import (
        aggregate_summary,
        load_aggregate,
        parse_summary_spec,
    )
except ImportError:
    from evaluation_qtl.summarize_qtl_benchmarks import (
        aggregate_summary,
        load_aggregate,
        parse_summary_spec,
    )


def example_summary():
    return pd.DataFrame(
        [
            {
                "epoch": 5,
                "aggregation": "fold_macro",
                "log_l2_pearson": 0.6,
                "fold_log_l2_pearson_median": 0.59,
                "fold_log_l2_pearson_iqr": 0.1,
                "fold_log_l2_pearson_min": 0.3,
                "n_folds": 9,
                "n_usable_folds": 9,
                "n_low_folds": 0,
                "low_fold_correlation_threshold": 0.2,
                "pred_log_l2_fold_mean_sd": 0.4,
                "pred_log_l2_fold_std_sd": 0.2,
            },
            {
                "epoch": 5,
                "aggregation": "fold_standardized_pooled",
                "log_l2_pearson": 0.58,
                "calibration_penalty": 0.12,
            },
            {
                "epoch": 5,
                "aggregation": "pooled_folds",
                "log_l2_pearson": 0.46,
            },
            {
                "epoch": 5,
                "aggregation": "fold0_ensemble",
                "log_l2_pearson": 0.48,
            },
            {
                "epoch": 5,
                "aggregation": "legacy_composite",
                "log_l2_pearson": 0.47,
            },
        ]
    )


class SummarizeQtlBenchmarksTest(unittest.TestCase):
    def test_parses_labeled_path(self):
        label, path = parse_summary_spec("clipnet=/tmp/summary.csv")
        self.assertEqual(label, "clipnet")
        self.assertEqual(path, Path("/tmp/summary.csv"))

    def test_pivots_aggregation_rows(self):
        records = aggregate_summary(
            example_summary(),
            "clipnet",
            Path("/tmp/tiqtl/epoch_qtl_benchmark_summary.csv"),
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["qtl"], "tiqtl")
        self.assertEqual(record["epoch"], 5)
        self.assertEqual(record["fold_macro_r"], 0.6)
        self.assertEqual(record["fold_standardized_pooled_r"], 0.58)
        self.assertEqual(record["pooled_folds_r"], 0.46)
        self.assertEqual(record["fold0_ensemble_r"], 0.48)
        self.assertEqual(record["legacy_composite_r"], 0.47)
        self.assertEqual(record["calibration_penalty"], 0.12)

    def test_calculates_missing_calibration_penalty(self):
        summary = example_summary().drop(columns="calibration_penalty")
        record = aggregate_summary(
            summary, "clipnet", Path("/tmp/summary.csv")
        )[0]
        self.assertAlmostEqual(record["calibration_penalty"], 0.12)

    def test_loads_multiple_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.csv"
            second = root / "second.csv"
            example_summary().to_csv(first, index=False)
            second_summary = example_summary()
            second_summary["epoch"] = 10
            second_summary.to_csv(second, index=False)

            result = load_aggregate([f"a={first}", f"b={second}"])

            self.assertEqual(result["label"].tolist(), ["a", "b"])
            self.assertEqual(result["epoch"].tolist(), [5, 10])
            self.assertTrue(np.isfinite(result["fold_macro_r"]).all())


if __name__ == "__main__":
    unittest.main()
