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
then reports the legacy composite Pearson correlation alongside the published
reference value (0.477 for tiQTLs or 0.542 for diQTLs). Outputs are isolated
under:

```text
PATH_TO_PREDICTIONS/{qtl}/published_clipnet_benchmark/{run_name}/
```
