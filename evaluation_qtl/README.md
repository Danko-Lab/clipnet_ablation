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
Pearson is retained as a separate diagnostic. Outputs are isolated under:

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
