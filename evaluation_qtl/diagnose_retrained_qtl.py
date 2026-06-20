#!/usr/bin/env python3

"""Diagnose retrained CLIPNET QTL benchmark results.

This script audits the pieces that can make retrained QTL benchmarks look odd:

- model-root layout and dataset parameter files
- expected sequence / PRO-cap array naming for each ablation
- validation-loss best epochs from training histories
- best-model QTL summary rows, fold pathologies, and calibration diagnostics

It only reads existing files and writes diagnostic tables.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUNS = ("clipnet", "mean_model", "ref_model")
FOLDS = tuple(range(1, 10))
QTL_SUMMARY_NAME = "best_model_qtl_benchmark_summary_published_l2.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models_root",
        type=Path,
        default=Path("../models"),
        help="Root containing clipnet/, mean_model/, and ref_model/ fold dirs.",
    )
    parser.add_argument(
        "--predictions_root",
        type=Path,
        default=Path("../predictions"),
        help="Root containing {qtl}/best_model_benchmark outputs.",
    )
    parser.add_argument(
        "--qtl",
        choices=["tiqtl", "diqtl"],
        default="tiqtl",
        help="QTL benchmark to inspect.",
    )
    parser.add_argument(
        "--runs",
        default=",".join(DEFAULT_RUNS),
        help=(
            "Comma-separated model run names to inspect. Use "
            "MODEL_RUN:PREDICTION_RUN when benchmark outputs were written "
            "under a different --run_name."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("retrained_qtl_diagnostics"),
        help="Directory for diagnostic tables.",
    )
    parser.add_argument(
        "--low_fold_threshold",
        type=float,
        default=0.2,
        help="Flag fold log-L2 Pearson values below this threshold.",
    )
    return parser.parse_args()


def run_specs(value):
    specs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            model_run, prediction_run = item.split(":", 1)
            specs.append(
                {
                    "run": model_run.strip(),
                    "prediction_run": prediction_run.strip(),
                    "label": prediction_run.strip(),
                }
            )
        else:
            specs.append({"run": item, "prediction_run": item, "label": item})
    return specs


def expected_patterns(run):
    if run == "ref_model":
        return {
            "seq": "concat_sequence_reference_",
            "procap": "concat_procap_",
        }
    if run == "mean_model":
        return {
            "seq": "concat_sequence_",
            "procap": "concat_mean_procap_",
        }
    return {
        "seq": "concat_sequence_",
        "procap": "concat_procap_",
    }


def fold_dir(models_root, run, fold):
    return models_root / run / f"f{fold}"


def path_list(params, key):
    value = params.get(key, [])
    return value if isinstance(value, list) else [value]


def audit_dataset_params(models_root, specs):
    rows = []
    for spec in specs:
        run = spec["run"]
        patterns = expected_patterns(run)
        for fold in FOLDS:
            directory = fold_dir(models_root, run, fold)
            params_path = directory / "dataset_params.json"
            row = {
                "label": spec["label"],
                "run": run,
                "prediction_run": spec["prediction_run"],
                "fold": fold,
                "fold_dir": str(directory),
                "dataset_params": str(params_path),
                "dataset_params_exists": params_path.exists(),
            }
            if not params_path.exists():
                rows.append(row)
                continue

            with params_path.open() as handle:
                params = json.load(handle)

            for key in ("train_seq", "val_seq", "test_seq"):
                paths = path_list(params, key)
                row[f"{key}_count"] = len(paths)
                row[f"{key}_pattern_ok"] = all(
                    patterns["seq"] in Path(path).name for path in paths
                )
                row[f"{key}_all_exist"] = all(Path(path).exists() for path in paths)
                row[f"{key}_example"] = paths[0] if paths else ""

            for key in ("train_procap", "val_procap", "test_procap"):
                paths = path_list(params, key)
                row[f"{key}_count"] = len(paths)
                row[f"{key}_pattern_ok"] = all(
                    patterns["procap"] in Path(path).name for path in paths
                )
                row[f"{key}_all_exist"] = all(Path(path).exists() for path in paths)
                row[f"{key}_example"] = paths[0] if paths else ""

            row["window_length"] = params.get("window_length")
            row["output_length"] = params.get("output_length")
            row["weight"] = params.get("weight")
            rows.append(row)
    return pd.DataFrame(rows)


def load_history(path):
    with path.open() as handle:
        data = json.load(handle)
    if "val_loss" not in data:
        return None
    return data


def find_history(directory):
    candidates = sorted(directory.glob("*history.json"))
    for candidate in candidates:
        history = load_history(candidate)
        if history is not None:
            return candidate, history
    return None, None


def audit_histories(models_root, specs):
    rows = []
    for spec in specs:
        run = spec["run"]
        for fold in FOLDS:
            directory = fold_dir(models_root, run, fold)
            history_path, history = find_history(directory)
            row = {
                "label": spec["label"],
                "run": run,
                "prediction_run": spec["prediction_run"],
                "fold": fold,
                "fold_dir": str(directory),
                "history_found": history_path is not None,
                "history_path": str(history_path) if history_path else "",
            }
            if history is None:
                rows.append(row)
                continue

            val_loss = np.asarray(history["val_loss"], dtype=float)
            best_index = int(np.nanargmin(val_loss))
            row.update(
                {
                    "n_epochs_recorded": len(val_loss),
                    "best_epoch": best_index + 1,
                    "best_val_loss": val_loss[best_index],
                    "last_epoch": len(val_loss),
                    "last_val_loss": val_loss[-1],
                    "best_is_last": best_index == len(val_loss) - 1,
                    "best_after_epoch_50": best_index + 1 > 50,
                    "last_minus_best_val_loss": val_loss[-1] - val_loss[best_index],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summary_candidates(predictions_root, qtl, run):
    root = predictions_root / qtl / "best_model_benchmark" / run
    return [
        root / "folds" / QTL_SUMMARY_NAME,
        root / QTL_SUMMARY_NAME,
    ]


def find_summary(predictions_root, qtl, run):
    for path in summary_candidates(predictions_root, qtl, run):
        if path.exists():
            return path
    return None


def audit_qtl_summaries(predictions_root, qtl, specs, low_fold_threshold):
    summary_rows = []
    fold_rows = []
    for spec in specs:
        run = spec["run"]
        prediction_run = spec["prediction_run"]
        label = spec["label"]
        path = find_summary(predictions_root, qtl, prediction_run)
        if path is None:
            summary_rows.append(
                {
                    "label": label,
                    "run": run,
                    "prediction_run": prediction_run,
                    "qtl": qtl,
                    "summary_found": False,
                    "summary_path": "",
                }
            )
            continue

        summary = pd.read_csv(path)
        for aggregation in (
            "fold_macro",
            "fold_standardized_pooled",
            "pooled_folds",
            "legacy_composite",
            "ensemble",
        ):
            rows = summary[summary["aggregation"] == aggregation]
            if rows.empty:
                continue
            row = rows.iloc[-1].to_dict()
            summary_rows.append(
                {
                    "run": run,
                    "label": label,
                    "prediction_run": prediction_run,
                    "qtl": qtl,
                    "summary_found": True,
                    "summary_path": str(path),
                    "aggregation": aggregation,
                    "log_l2_pearson": row.get("log_l2_pearson"),
                    "n_snps": row.get("n_snps"),
                    "n_usable_folds": row.get("n_usable_folds"),
                    "n_low_folds": row.get("n_low_folds"),
                    "fold_log_l2_pearson_min": row.get(
                        "fold_log_l2_pearson_min"
                    ),
                    "fold_log_l2_pearson_iqr": row.get(
                        "fold_log_l2_pearson_iqr"
                    ),
                    "pred_log_l2_fold_mean_sd": row.get(
                        "pred_log_l2_fold_mean_sd"
                    ),
                    "pred_log_l2_fold_std_sd": row.get(
                        "pred_log_l2_fold_std_sd"
                    ),
                    "calibration_penalty": row.get("calibration_penalty"),
                }
            )

        folds = summary[summary["aggregation"] == "fold"].copy()
        for _, row in folds.iterrows():
            fold_rows.append(
                {
                    "run": run,
                    "label": label,
                    "prediction_run": prediction_run,
                    "qtl": qtl,
                    "summary_path": str(path),
                    "fold": row.get("fold"),
                    "n_snps": row.get("n_snps"),
                    "log_l2_pearson": row.get("log_l2_pearson"),
                    "is_low_fold": row.get("log_l2_pearson") < low_fold_threshold,
                    "pred_log_l2_mean": row.get("pred_log_l2_mean"),
                    "pred_log_l2_std": row.get("pred_log_l2_std"),
                    "expt_log_l2_mean": row.get("expt_log_l2_mean"),
                    "expt_log_l2_std": row.get("expt_log_l2_std"),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def write_issues(output_dir, dataset, histories, summaries, folds):
    issues = []
    missing_params = dataset[~dataset["dataset_params_exists"]]
    for _, row in missing_params.iterrows():
        issues.append(f"Missing dataset_params.json for {row.run} fold {row.fold}.")

    for _, row in dataset[dataset["dataset_params_exists"]].iterrows():
        for column in dataset.columns:
            if column.endswith("_pattern_ok") and row.get(column) is False:
                issues.append(f"{row.run} fold {row.fold}: {column} is false.")
            if column.endswith("_all_exist") and row.get(column) is False:
                issues.append(f"{row.run} fold {row.fold}: {column} is false.")

    missing_histories = histories[~histories["history_found"]]
    for _, row in missing_histories.iterrows():
        issues.append(f"Missing history JSON for {row.run} fold {row.fold}.")

    if "best_is_last" in histories:
        for _, row in histories[histories["best_is_last"].fillna(False)].iterrows():
            issues.append(
                f"{row.run} fold {row.fold}: validation minimum is the last "
                f"recorded epoch ({int(row.best_epoch)})."
            )

    for _, row in summaries[~summaries["summary_found"]].iterrows():
        issues.append(f"Missing QTL summary for {row.run} {row.qtl}.")

    if not folds.empty:
        for _, row in folds[folds["is_low_fold"]].iterrows():
            issues.append(
                f"{row.run} {row.qtl} fold {row.fold}: low fold Pearson "
                f"{row.log_l2_pearson:.3f}."
            )

    issues_path = output_dir / "issues.txt"
    issues_path.write_text("\n".join(issues) + ("\n" if issues else ""))
    return issues_path


def main():
    args = parse_args()
    specs = run_specs(args.runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = audit_dataset_params(args.models_root, specs)
    histories = audit_histories(args.models_root, specs)
    summaries, folds = audit_qtl_summaries(
        args.predictions_root, args.qtl, specs, args.low_fold_threshold
    )

    dataset.to_csv(args.output_dir / "dataset_params_audit.csv", index=False)
    histories.to_csv(args.output_dir / "history_audit.csv", index=False)
    summaries.to_csv(args.output_dir / "qtl_summary_audit.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    issues_path = write_issues(args.output_dir, dataset, histories, summaries, folds)

    print(f"Wrote diagnostics to {args.output_dir}")
    print(f"  dataset params: {args.output_dir / 'dataset_params_audit.csv'}")
    print(f"  histories:      {args.output_dir / 'history_audit.csv'}")
    print(f"  QTL summaries:  {args.output_dir / 'qtl_summary_audit.csv'}")
    print(f"  fold metrics:   {args.output_dir / 'fold_metrics.csv'}")
    print(f"  issues:         {issues_path}")


if __name__ == "__main__":
    main()
