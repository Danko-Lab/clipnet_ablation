import unittest

import numpy as np
import pandas as pd

try:
    from fold_calibrated_scoring import build_fold_calibrated_rows
except ImportError:
    from evaluation_qtl.fold_calibrated_scoring import build_fold_calibrated_rows


def log_values(values):
    return np.log(np.asarray(values) + 1e-3)


def correlation(x, y, method):
    x = pd.Series(x)
    y = pd.Series(y)
    if method == "spearman":
        x = x.rank()
        y = y.rank()
    return x.corr(y, method="pearson")


def summarize(scores, fold, aggregation):
    expt = log_values(scores["expt"])
    pred = log_values(scores["pred"])
    return {
        "aggregation": aggregation,
        "fold": fold,
        "n_snps": len(scores),
        "n_valid_snps": len(scores),
        "n_invalid_snps": 0,
        "l2_pearson": correlation(scores["expt"], scores["pred"], "pearson"),
        "l2_spearman": correlation(scores["expt"], scores["pred"], "spearman"),
        "l2_pearson_valid": correlation(scores["expt"], scores["pred"], "pearson"),
        "l2_spearman_valid": correlation(scores["expt"], scores["pred"], "spearman"),
        "log_l2_pearson": correlation(expt, pred, "pearson"),
        "log_l2_pearson_valid": correlation(expt, pred, "pearson"),
        "manuscript_pearson": correlation(expt, pred, "pearson"),
        "pred_l2_mean": scores["pred"].mean(),
        "pred_l2_std": scores["pred"].std(),
        "expt_l2_mean": scores["expt"].mean(),
        "expt_l2_std": scores["expt"].std(),
        "pred_log_l2_mean": pred.mean(),
        "pred_log_l2_std": pred.std(ddof=1),
        "expt_log_l2_mean": expt.mean(),
        "expt_log_l2_std": expt.std(ddof=1),
    }


class FoldCalibratedScoringTest(unittest.TestCase):
    def test_macro_statistics_and_low_fold_count(self):
        fold_scores = {
            1: pd.DataFrame({"expt": [1, 2, 4, 8], "pred": [1, 2, 4, 8]}),
            2: pd.DataFrame({"expt": [1, 2, 4, 8], "pred": [8, 4, 2, 1]}),
            0: pd.DataFrame({"expt": [1, 2, 3], "pred": [1, 2, 3]}),
        }
        pooled_row = summarize(
            pd.concat([fold_scores[1], fold_scores[2]]), "pooled", "pooled_folds"
        )

        rows = build_fold_calibrated_rows(
            fold_scores,
            summarize,
            log_values,
            correlation,
            pooled_row,
        )
        macro = rows[0]
        fold_rs = np.asarray(
            [
                correlation(
                    log_values(scores["expt"]),
                    log_values(scores["pred"]),
                    "pearson",
                )
                for scores in (fold_scores[1], fold_scores[2])
            ]
        )
        expected_macro = np.tanh(
            np.mean(
                np.arctanh(np.clip(fold_rs, -1 + 1e-12, 1 - 1e-12))
            )
        )

        self.assertEqual(macro["aggregation"], "fold_macro")
        self.assertEqual(macro["n_folds"], 2)
        self.assertEqual(macro["n_usable_folds"], 2)
        self.assertEqual(macro["n_low_folds"], 1)
        self.assertAlmostEqual(
            macro["fold_log_l2_pearson_median"], np.median(fold_rs)
        )
        self.assertAlmostEqual(
            macro["fold_log_l2_pearson_iqr"],
            np.percentile(fold_rs, 75) - np.percentile(fold_rs, 25),
        )
        self.assertAlmostEqual(
            macro["fold_log_l2_pearson_min"], np.min(fold_rs)
        )
        self.assertAlmostEqual(macro["fold_fisher_macro_r"], expected_macro)

    def test_standardization_removes_fold_offsets_and_scales(self):
        base = np.array([1.0, 2.0, 4.0, 8.0])
        fold_scores = {
            1: pd.DataFrame({"expt": base, "pred": base}),
            2: pd.DataFrame({"expt": base * 100, "pred": base * 0.01}),
        }
        pooled = pd.concat(fold_scores.values())
        pooled_row = summarize(pooled, "pooled", "pooled_folds")

        rows = build_fold_calibrated_rows(
            fold_scores, summarize, log_values, correlation, pooled_row
        )
        standardized = rows[1]

        self.assertEqual(
            standardized["aggregation"], "fold_standardized_pooled"
        )
        self.assertGreater(standardized["log_l2_pearson"], 0.999)
        self.assertAlmostEqual(
            standardized["calibration_penalty"],
            standardized["log_l2_pearson"] - pooled_row["log_l2_pearson"],
        )

    def test_nonfinite_and_constant_folds_are_excluded(self):
        fold_scores = {
            1: pd.DataFrame({"expt": [1, 2, 4], "pred": [1, 2, 4]}),
            2: pd.DataFrame({"expt": [1, 1, 1], "pred": [1, 2, 3]}),
            3: pd.DataFrame({"expt": [1, 2, 3], "pred": [1, np.nan, 3]}),
        }
        pooled_row = summarize(fold_scores[1], "pooled", "pooled_folds")

        rows = build_fold_calibrated_rows(
            fold_scores, summarize, log_values, correlation, pooled_row
        )

        self.assertEqual(rows[0]["n_folds"], 3)
        self.assertEqual(rows[0]["n_usable_folds"], 1)
        self.assertEqual(rows[1]["n_snps"], 3)


if __name__ == "__main__":
    unittest.main()
