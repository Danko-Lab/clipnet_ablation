"""Fold-robust summary statistics for QTL benchmark score tables."""

import numpy as np
import pandas as pd


LOW_FOLD_CORRELATION_THRESHOLD = 0.2
RAW_METRIC_COLUMNS = (
    "l2_pearson",
    "l2_spearman",
    "l2_pearson_valid",
    "l2_spearman_valid",
    "pred_l2_mean",
    "pred_l2_std",
    "expt_l2_mean",
    "expt_l2_std",
)


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else np.nan


def _usable_log_scores(scores, log_transform):
    expt = log_transform(scores["expt"].to_numpy())
    pred = log_transform(scores["pred"].to_numpy())
    if (
        len(expt) < 2
        or not np.isfinite(expt).all()
        or not np.isfinite(pred).all()
        or np.std(expt) == 0
        or np.std(pred) == 0
    ):
        return None
    return expt, pred


def build_fold_calibrated_rows(
    fold_scores,
    summarize,
    log_transform,
    correlation,
    pooled_row,
    low_threshold=LOW_FOLD_CORRELATION_THRESHOLD,
):
    """Build equal-fold Fisher and within-fold-standardized summary rows.

    ``fold_scores`` maps independently held-out fold IDs to per-SNP score
    DataFrames. ``summarize`` must have the signature
    ``summarize(scores, fold, aggregation)``.
    """
    fold_statistics = []
    standardized_parts = []
    usable_scores = []
    independent_fold_count = sum(str(fold) != "0" for fold in fold_scores)

    for fold, scores in fold_scores.items():
        if str(fold) == "0":
            continue
        logged = _usable_log_scores(scores, log_transform)
        if logged is None:
            continue
        expt_log, pred_log = logged
        fold_r = correlation(expt_log, pred_log, "pearson")
        if not np.isfinite(fold_r):
            continue

        fold_statistics.append(
            {
                "fold": fold,
                "r": float(fold_r),
                "pred_mean": float(np.mean(pred_log)),
                "pred_std": _sample_std(pred_log),
                "n_snps": len(scores),
            }
        )
        standardized_parts.append(
            pd.DataFrame(
                {
                    "expt": (expt_log - np.mean(expt_log)) / np.std(expt_log, ddof=1),
                    "pred": (pred_log - np.mean(pred_log)) / np.std(pred_log, ddof=1),
                },
                index=scores.index,
            )
        )
        usable_scores.append(scores)

    if not fold_statistics:
        return []

    correlations = np.asarray([item["r"] for item in fold_statistics])
    fisher_macro = float(
        np.tanh(np.mean(np.arctanh(np.clip(correlations, -1 + 1e-12, 1 - 1e-12))))
    )
    pred_means = np.asarray([item["pred_mean"] for item in fold_statistics])
    pred_stds = np.asarray([item["pred_std"] for item in fold_statistics])
    pooled_usable = pd.concat(usable_scores)

    macro_row = summarize(pooled_usable, "macro", "fold_macro")
    macro_row.update(
        {
            "log_l2_pearson": fisher_macro,
            "log_l2_pearson_valid": fisher_macro,
            "manuscript_pearson": fisher_macro,
            "fold_fisher_macro_r": fisher_macro,
            "fold_log_l2_pearson_median": float(np.median(correlations)),
            "fold_log_l2_pearson_iqr": float(
                np.percentile(correlations, 75) - np.percentile(correlations, 25)
            ),
            "fold_log_l2_pearson_min": float(np.min(correlations)),
            "n_folds": independent_fold_count,
            "n_usable_folds": len(fold_statistics),
            "n_low_folds": int(np.sum(correlations < low_threshold)),
            "low_fold_correlation_threshold": low_threshold,
            "pred_log_l2_fold_mean_sd": _sample_std(pred_means),
            "pred_log_l2_fold_std_sd": _sample_std(pred_stds),
            "calibration_penalty": np.nan,
        }
    )
    for column in RAW_METRIC_COLUMNS:
        macro_row[column] = np.nan

    standardized = pd.concat(standardized_parts)
    standardized_r = correlation(
        standardized["expt"].to_numpy(),
        standardized["pred"].to_numpy(),
        "pearson",
    )
    standardized_row = summarize(
        pooled_usable, "standardized", "fold_standardized_pooled"
    )
    standardized_row.update(
        {
            "n_snps": standardized.shape[0],
            "n_valid_snps": standardized.shape[0],
            "n_invalid_snps": 0,
            "log_l2_pearson": standardized_r,
            "log_l2_pearson_valid": standardized_r,
            "manuscript_pearson": standardized_r,
            "pred_log_l2_mean": float(standardized["pred"].mean()),
            "pred_log_l2_std": float(standardized["pred"].std(ddof=1)),
            "expt_log_l2_mean": float(standardized["expt"].mean()),
            "expt_log_l2_std": float(standardized["expt"].std(ddof=1)),
            "fold_fisher_macro_r": fisher_macro,
            "fold_log_l2_pearson_median": float(np.median(correlations)),
            "fold_log_l2_pearson_iqr": float(
                np.percentile(correlations, 75) - np.percentile(correlations, 25)
            ),
            "fold_log_l2_pearson_min": float(np.min(correlations)),
            "n_folds": independent_fold_count,
            "n_usable_folds": len(fold_statistics),
            "n_low_folds": int(np.sum(correlations < low_threshold)),
            "low_fold_correlation_threshold": low_threshold,
            "pred_log_l2_fold_mean_sd": _sample_std(pred_means),
            "pred_log_l2_fold_std_sd": _sample_std(pred_stds),
            "calibration_penalty": (
                standardized_r - pooled_row["log_l2_pearson"]
                if pooled_row is not None
                and np.isfinite(pooled_row["log_l2_pearson"])
                else np.nan
            ),
        }
    )
    for column in RAW_METRIC_COLUMNS:
        standardized_row[column] = np.nan

    return [macro_row, standardized_row]
