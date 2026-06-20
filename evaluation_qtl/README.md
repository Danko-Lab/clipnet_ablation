# QTL prediction benchmark

This directory contains notebooks for evaluating the ablated models at predicting QTL effects. The notebooks contained in the base directory will generate the plots used in the paper.

Python scripts in `diqtl_predict/` and `tiqtl_predict/` were used to actually generate the predictions, but these still need quite a bit of cleaning up.

## Published CLIPNET benchmark

`benchmark_published_clipnet.py` runs the current QTL pipeline against the
published models from [Zenodo record 10408623](https://zenodo.org/records/10408623).
Download `fold_1.h5` through `fold_9.h5` and
`data_fold_assignments.csv` into one directory, then run:

```bash
python evaluation_qtl/benchmark_published_clipnet.py all PATH_TO_MODELS \
  --qtl tiqtl \
  --mode composite \
  --data_root PATH_TO_DATA \
  --qtl_data_dir PATH_TO_DATA/tiqtl \
  --predictions_root PATH_TO_PREDICTIONS \
  --gpu
```

Repeat with `--qtl diqtl` for the divergent-initiation benchmark. Composite
mode creates both the across-model ensemble and the held-out-fold predictions,
then reports the legacy composite log-L2 Pearson correlation alongside the
published reference value (0.477 for tiQTLs or 0.542 for diQTLs). Raw-L2
Pearson is retained as a separate diagnostic. The fold summary also includes
`fold_macro`, `fold_standardized_pooled`, `pooled_folds`, and the calibration
penalty, using the same definitions as the ablation benchmarks. Outputs are
isolated under:

```text
PATH_TO_PREDICTIONS/{qtl}/published_clipnet_benchmark/{run_name}/
```

## Published experimental L2 targets

All three benchmark entrypoints can score model predictions against the
canonical observed per-SNP L2 values in the official `qtl_analysis.tar.gz`:

```bash
python evaluation_qtl/benchmark_published_clipnet.py score PATH_TO_MODELS \
  --qtl diqtl \
  --mode composite \
  --data_root PATH_TO_DATA \
  --qtl_data_dir PATH_TO_DATA/diqtl \
  --predictions_root PATH_TO_PREDICTIONS \
  --experimental_l2_archive PATH_TO/qtl_analysis.tar.gz
```

The same option works with `benchmark_best_model.py` and
`benchmark_epoch_checkpoints.py`. It reuses existing split prediction joblibs,
joins predicted L2 values to canonical experimental L2 values by SNP ID, and
does not require prediction or split regeneration.

Archive-backed outputs are kept separate:

```text
scores_published_l2/
best_model_qtl_benchmark_summary_published_l2.csv
epoch_qtl_benchmark_summary_published_l2.csv
published_clipnet_qtl_benchmark_summary_published_l2.csv
```

Summary rows record `experimental_source=published_archive`. Without the
option, benchmarks continue to use generated PRO-cap tracks and write their
existing output paths.

To add fold-calibrated rows to an existing published benchmark without
regenerating predictions or allele splits, rerun only scoring:

```bash
python evaluation_qtl/benchmark_published_clipnet.py score PATH_TO_MODELS \
  --qtl tiqtl \
  --mode composite \
  --data_root PATH_TO_DATA \
  --qtl_data_dir PATH_TO_DATA/tiqtl \
  --predictions_root PATH_TO_PREDICTIONS \
  --experimental_l2_archive PATH_TO/qtl_analysis.tar.gz
```

Repeat with `--qtl diqtl`. The command rewrites the published summary CSV and
prints the fold macro, standardized pooled, ordinary pooled, calibration
penalty, and usable-fold count.

## Fold-calibrated summaries

Fold-mode scoring for best-model, epoch-checkpoint, and published-model
benchmarks also writes:

- `fold_macro`: the primary model-comparison statistic, using an equal-fold
  Fisher average of held-out fold log-L2 Pearson correlations.
- `fold_standardized_pooled`: z-scores observed and predicted log-L2 values
  within each held-out fold before pooling, removing fold-specific offsets and
  scales.
- `pooled_folds`: retains sensitivity to cross-fold calibration.
- `legacy_composite`: retains manuscript comparability by adding the fold-0
  ensemble remainder.

The summary includes fold median, IQR, minimum, folds below `r=0.2`,
across-fold prediction calibration variability, and
`calibration_penalty = fold_standardized_pooled - pooled_folds`. These rows use
folds 1-9 only; fold 0 is never included in the fold-calibrated statistics.

## Aggregate reports and plots

Combine best-model and epoch summaries from multiple ablations into one wide
CSV and a set of PNG/PDF plots:

```bash
python evaluation_qtl/summarize_qtl_benchmarks.py \
  clipnet=PATH_TO_CLIPNET/epoch_qtl_benchmark_summary_published_l2.csv \
  ref_model=PATH_TO_REF/epoch_qtl_benchmark_summary_published_l2.csv \
  mean_model=PATH_TO_MEAN/epoch_qtl_benchmark_summary_published_l2.csv \
  clipnet_best=PATH_TO_CLIPNET/best_model_qtl_benchmark_summary_published_l2.csv \
  --output_dir PATH_TO_REPORT
```

The aggregate table has one row per model checkpoint or training epoch. It
places `fold_macro`, fold-standardized pooled, pooled-fold, and legacy
correlations in adjacent columns along with fold stability and calibration
diagnostics. Use `--no_plots` when Matplotlib is unavailable.

## Retrained-model diagnostics

Use `diagnose_retrained_qtl.py` when retrained ablation results look odd. It
audits model fold directories, `dataset_params.json` path patterns, training
history best epochs, QTL summary rows, low-performing folds, and calibration
diagnostics:

```bash
python evaluation_qtl/diagnose_retrained_qtl.py \
  --models_root ../models \
  --predictions_root ../predictions \
  --qtl tiqtl \
  --output_dir evaluation_qtl/tiqtl_retrain_diagnostics
```

If the benchmark was written with a fresh `--run_name`, map the model directory
name to that prediction namespace:

```bash
python evaluation_qtl/diagnose_retrained_qtl.py \
  --models_root ../models \
  --predictions_root ../predictions \
  --qtl tiqtl \
  --runs clipnet:clipnet_2gpu_retrain,mean_model:mean_model_2gpu_retrain,ref_model:ref_model_2gpu_retrain \
  --output_dir evaluation_qtl/tiqtl_retrain_diagnostics
```

The companion command file `retrained_diagnostic_commands.sh` lists the full
remote-server workflow for regenerating fold-mode best-model summaries,
published CLIPNET controls, diagnostics, and aggregate plots in clean output
namespaces.


time python evaluation_qtl/summarize_qtl_benchmarks.py \
  published=predictions/tiqtl/published_clipnet_benchmark/zenodo_10408623/folds/published_clipnet_qtl_benchmark_summary_published_l2.csv \
  clipnet=predictions/tiqtl/epoch_checkpoint_benchmark/clipnet/folds/epoch_qtl_benchmark_summary_published_l2.csv \
  ref_model=predictions/tiqtl/epoch_checkpoint_benchmark/ref_model/folds/epoch_qtl_benchmark_summary_published_l2.csv \
  mean_model=predictions/tiqtl/epoch_checkpoint_benchmark/mean_model/folds/epoch_qtl_benchmark_summary_published_l2.csv \
  clipnet_best=predictions/tiqtl/best_model_benchmark/clipnet/folds/best_model_qtl_benchmark_summary_published_l2.csv \
  ref_best=predictions/tiqtl/best_model_benchmark/ref_model/folds/best_model_qtl_benchmark_summary_published_l2.csv \
  mean_best=predictions/tiqtl/best_model_benchmark/mean_model/folds/best_model_qtl_benchmark_summary_published_l2.csv \
  --output_dir predictions/tiqtl_summaries/

time python evaluation_qtl/summarize_qtl_benchmarks.py \
  published=predictions/diqtl/published_clipnet_benchmark/zenodo_10408623/folds/published_clipnet_qtl_benchmark_summary_published_l2.csv \
  clipnet=predictions/diqtl/epoch_checkpoint_benchmark/clipnet/folds/epoch_qtl_benchmark_summary_published_l2.csv \
  ref_model=predictions/diqtl/epoch_checkpoint_benchmark/ref_model/folds/epoch_qtl_benchmark_summary_published_l2.csv \
  mean_model=predictions/diqtl/epoch_checkpoint_benchmark/mean_model/folds/epoch_qtl_benchmark_summary_published_l2.csv \
  clipnet_best=predictions/diqtl/best_model_benchmark/clipnet/folds/best_model_qtl_benchmark_summary_published_l2.csv \
  ref_best=predictions/diqtl/best_model_benchmark/ref_model/folds/best_model_qtl_benchmark_summary_published_l2.csv \
  mean_best=predictions/diqtl/best_model_benchmark/mean_model/folds/best_model_qtl_benchmark_summary_published_l2.csv \
  --output_dir predictions/diqtl_summaries/
