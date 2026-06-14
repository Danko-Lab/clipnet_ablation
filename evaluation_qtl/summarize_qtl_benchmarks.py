#!/usr/bin/env python3

"""Aggregate QTL benchmark summaries and plot model/epoch comparisons.

Inputs may be supplied as paths or as LABEL=PATH pairs. The script reads the
existing benchmark summary CSVs; it does not rerun prediction or scoring.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


AGGREGATION_COLUMNS = {
    "fold_macro": "fold_macro_r",
    "fold_standardized_pooled": "fold_standardized_pooled_r",
    "pooled_folds": "pooled_folds_r",
    "legacy_composite": "legacy_composite_r",
    "ensemble": "ensemble_r",
}
DIAGNOSTIC_COLUMNS = (
    "fold_log_l2_pearson_median",
    "fold_log_l2_pearson_iqr",
    "fold_log_l2_pearson_min",
    "n_folds",
    "n_usable_folds",
    "n_low_folds",
    "low_fold_correlation_threshold",
    "pred_log_l2_fold_mean_sd",
    "pred_log_l2_fold_std_sd",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        nargs="+",
        help="Benchmark summary CSV paths, optionally written as LABEL=PATH.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("qtl_benchmark_report"),
        help="Directory for the aggregate table and plots.",
    )
    parser.add_argument(
        "--output_prefix",
        default="qtl_benchmark",
        help="Prefix for generated output files.",
    )
    parser.add_argument(
        "--title",
        default="QTL benchmark comparison",
        help="Figure title.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Write the aggregate table without importing Matplotlib.",
    )
    return parser.parse_args()


def default_label(path):
    parent = path.parent
    if parent.name == "folds":
        parent = parent.parent
    return parent.name or path.stem


def parse_summary_spec(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
        if not label or not path:
            raise ValueError(f"Invalid LABEL=PATH summary specification: {spec}")
        return label, Path(path)
    path = Path(spec)
    return default_label(path), path


def infer_qtl(path):
    for part in reversed(path.parts):
        normalized = part.lower().replace("-", "").replace("_", "")
        if normalized in {"tiqtl", "tiqtls"}:
            return "tiqtl"
        if normalized in {"diqtl", "diqtls"}:
            return "diqtl"
    return ""


def infer_benchmark(path):
    name = path.name
    if name.startswith("epoch_"):
        return "epoch"
    if name.startswith("best_model_"):
        return "best_model"
    if name.startswith("published_clipnet_"):
        return "published_clipnet"
    return re.sub(r"_qtl_benchmark_summary.*$", "", path.stem)


def first_aggregation(group, aggregation):
    rows = group[group["aggregation"] == aggregation]
    if rows.empty:
        return None
    return rows.iloc[-1]


def aggregate_summary(summary, label, path):
    required = {"aggregation", "log_l2_pearson"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if "epoch" in summary.columns and summary["epoch"].notna().any():
        grouped = summary.groupby("epoch", dropna=False, sort=True)
    else:
        grouped = [(np.nan, summary)]

    records = []
    for epoch, group in grouped:
        record = {
            "label": label,
            "qtl": infer_qtl(path),
            "benchmark": infer_benchmark(path),
            "checkpoint": (
                group["checkpoint"].dropna().iloc[-1]
                if "checkpoint" in group and group["checkpoint"].notna().any()
                else ("epoch" if pd.notna(epoch) else infer_benchmark(path))
            ),
            "epoch": int(epoch) if pd.notna(epoch) else np.nan,
            "source_file": str(path),
        }
        for aggregation, output_column in AGGREGATION_COLUMNS.items():
            row = first_aggregation(group, aggregation)
            record[output_column] = (
                row["log_l2_pearson"] if row is not None else np.nan
            )

        macro = first_aggregation(group, "fold_macro")
        standardized = first_aggregation(group, "fold_standardized_pooled")
        for column in DIAGNOSTIC_COLUMNS:
            record[column] = (
                macro[column]
                if macro is not None and column in macro.index
                else np.nan
            )
        record["calibration_penalty"] = (
            standardized["calibration_penalty"]
            if standardized is not None
            and "calibration_penalty" in standardized.index
            else (
                record["fold_standardized_pooled_r"] - record["pooled_folds_r"]
                if pd.notna(record["fold_standardized_pooled_r"])
                and pd.notna(record["pooled_folds_r"])
                else np.nan
            )
        )
        records.append(record)
    return records


def load_aggregate(specs):
    records = []
    seen_labels = set()
    for spec in specs:
        label, path = parse_summary_spec(spec)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark summary does not exist: {path}")
        key = (label, str(path))
        if key in seen_labels:
            raise ValueError(f"Duplicate benchmark input: {label}={path}")
        seen_labels.add(key)
        records.extend(aggregate_summary(pd.read_csv(path), label, path))

    result = pd.DataFrame(records)
    return result.sort_values(
        ["qtl", "benchmark", "label", "epoch"], na_position="last"
    ).reset_index(drop=True)


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires Matplotlib. Install it or rerun with --no_plots."
        ) from exc
    return plt


def save_figure(fig, output_dir, stem):
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{extension}",
            dpi=200,
            bbox_inches="tight",
        )


def plot_best_models(data, output_dir, prefix, title, plt):
    best = data[data["epoch"].isna()].copy()
    if best.empty:
        return []

    metrics = [
        ("fold_macro_r", "Fold macro"),
        ("fold_standardized_pooled_r", "Fold-standardized"),
        ("pooled_folds_r", "Pooled folds"),
        ("legacy_composite_r", "Legacy composite"),
    ]
    x = np.arange(len(best))
    width = 0.19
    fig, axis = plt.subplots(figsize=(max(7, len(best) * 1.15), 4.8))
    for index, (column, display) in enumerate(metrics):
        axis.bar(
            x + (index - 1.5) * width,
            best[column],
            width,
            label=display,
        )
    axis.set_xticks(x, best["label"], rotation=30, ha="right")
    axis.set_ylabel("Log-L2 Pearson correlation")
    axis.set_title(f"{title}: best checkpoints")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    stem = f"{prefix}_best_models"
    save_figure(fig, output_dir, stem)
    plt.close(fig)
    return [stem]


def plot_epochs(data, output_dir, prefix, title, plt):
    epochs = data[data["epoch"].notna()].copy()
    if epochs.empty:
        return []

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)
    for label, group in epochs.groupby("label", sort=False):
        group = group.sort_values("epoch")
        axes[0].plot(
            group["epoch"],
            group["fold_macro_r"],
            marker="o",
            label=label,
        )
        axes[1].plot(
            group["epoch"],
            group["calibration_penalty"],
            marker="o",
            label=label,
        )
    axes[0].set_ylabel("Fold macro log-L2 Pearson")
    axes[0].set_title(f"{title}: training epochs")
    axes[1].set_ylabel("Calibration penalty")
    axes[1].set_xlabel("Epoch")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.tight_layout()
    stem = f"{prefix}_epochs"
    save_figure(fig, output_dir, stem)
    plt.close(fig)
    return [stem]


def plot_calibration(data, output_dir, prefix, title, plt):
    usable = data[data["fold_macro_r"].notna()].copy()
    if usable.empty:
        return []
    usable["display"] = usable["label"]
    has_epoch = usable["epoch"].notna()
    usable.loc[has_epoch, "display"] = (
        usable.loc[has_epoch, "label"]
        + " e"
        + usable.loc[has_epoch, "epoch"].astype(int).astype(str)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, len(usable) * 0.32)))
    y = np.arange(len(usable))
    axes[0].errorbar(
        usable["fold_macro_r"],
        y,
        xerr=usable["fold_log_l2_pearson_iqr"] / 2,
        fmt="o",
        capsize=3,
    )
    axes[0].scatter(
        usable["fold_log_l2_pearson_min"],
        y,
        marker="|",
        s=100,
        label="Fold minimum",
    )
    axes[0].set_yticks(y, usable["display"])
    axes[0].set_xlabel("Fold macro correlation (IQR / minimum)")
    axes[0].legend(frameon=False)

    axes[1].scatter(
        usable["pred_log_l2_fold_mean_sd"],
        usable["calibration_penalty"],
    )
    for _, row in usable.iterrows():
        axes[1].annotate(
            row["display"],
            (row["pred_log_l2_fold_mean_sd"], row["calibration_penalty"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("Across-fold SD of predicted log-L2 means")
    axes[1].set_ylabel("Calibration penalty")
    axes[1].axhline(0, color="black", linewidth=0.7)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f"{title}: fold stability and calibration")
    fig.tight_layout()
    stem = f"{prefix}_calibration"
    save_figure(fig, output_dir, stem)
    plt.close(fig)
    return [stem]


def main():
    args = parse_args()
    data = load_aggregate(args.summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / f"{args.output_prefix}_stats.csv"
    data.to_csv(table_path, index=False, float_format="%.6g")
    print(f"Saved aggregate statistics to {table_path}")

    if args.no_plots:
        return
    plt = import_pyplot()
    stems = []
    qtl_values = [value for value in data["qtl"].drop_duplicates()]
    for qtl in qtl_values:
        subset = data[data["qtl"] == qtl]
        suffix = f"_{qtl}" if qtl else ""
        plot_prefix = f"{args.output_prefix}{suffix}"
        plot_title = f"{args.title} ({qtl})" if qtl else args.title
        stems.extend(
            plot_best_models(
                subset, args.output_dir, plot_prefix, plot_title, plt
            )
        )
        stems.extend(
            plot_epochs(subset, args.output_dir, plot_prefix, plot_title, plt)
        )
        stems.extend(
            plot_calibration(
                subset, args.output_dir, plot_prefix, plot_title, plt
            )
        )
    if stems:
        print(
            "Saved plots: "
            + ", ".join(str(args.output_dir / f"{stem}.png") for stem in stems)
        )
    else:
        print("No plottable fold-calibrated rows were found.")


if __name__ == "__main__":
    main()
