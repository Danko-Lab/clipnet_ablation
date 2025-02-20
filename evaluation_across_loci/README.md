# Evaluation of ablated models

This directory contains notebooks for plotting prediction performance of the ablated models. This directory only covers evaluations across genomic loci (i.e., can the models reconstruct PRO-cap tracks from the reference genome). For variant effect performance, see `../evaluation_qtl/`.

`plot_all_performances.ipynb` generates the main figure panels evaluating all the models.

`personalized_v_reference_predictions.ipynb` generates figure panels doing paired comparisons (CLIPNET(X) vs CLIPNET_reference(X)) of the personalized and reference-trained CLIPNET models.

Below I've sketched out how to generate these predictions, but this mess of a codebase should be rewritten to directly reference our Zenodo accessions.

## Prediction commands

```python
import os
import itertools
from pathlib import Path

clipnet_install = "/home2/ayh8/clipnet"
out_dir = Path("/home2/ayh8/predictions/clipnet_subsampling")
out_dir.mkdir(exist_ok=True, parents=True)
fasta_fp = Path("/home2/ayh8/data/lcl/fixed_windows/").joinpath("merged_sequence_0.fna.gz")
procap_fp = Path("/home2/ayh8/data/lcl/fixed_windows/").joinpath("merged_procap_0.npz")

n_individuals = [1, 5, 10, 15, 20, 30]
runs = range(5)

for n, r in itertools.product(n_individuals, runs):
    run = f"n{n}_run{r}"
    model_dir = Path("/home2/ayh8/clipnet_subsampling/models/").joinpath(run)
    out_fp = out_dir.joinpath(f"merged_{run}_fold_0_predictions.h5")
    if not os.path.exists(out_fp):
        cmd = f"clipnet predict -f {fasta_fp} -o {out_fp} -m {str(model_dir)}"
        print(cmd)
        os.system(cmd)
    else:
        print(f"{out_fp} exists, skipping.")

for run in runs:
    model_dir = Path("/home2/ayh8/clipnet_subsampling/models/").joinpath(run)
    for fold in range(1, 10):
        out_fp = out_dir.joinpath(f"fold_{fold}_{run}_fold_0_predictions.h5")
        cmd = f"python {clipnet_install}/calculate_performance_metrics.py {model_dir.joinpath(f'f{fold}.h5')} {procap_fp} {out_fp}"
        print(cmd)
        os.system(cmd)
```

## Performance metrics

```python
import os
import itertools
from pathlib import Path

clipnet_install = "/home2/ayh8/clipnet"
out_dir = Path("/home2/ayh8/predictions/clipnet_subsampling/")
out_dir.mkdir(exist_ok=True, parents=True)
procap_fp = Path("/home2/ayh8/data/lcl/fixed_windows/").joinpath("merged_procap_0.csv.gz")

n_individuals = [1, 5, 10, 15, 20, 30]
runs = range(5)

for n, r in itertools.product(n_individuals, runs):
    run = f"n{n}_run{r}"
    predict_fp = out_dir.joinpath(f"merged_{run}_fold_0_predictions.h5")
    out_fp = out_dir.joinpath(f"merged_{run}_fold_0_performance_metrics.h5")
    if not os.path.exists(out_fp):
        cmd = f"python {clipnet_install}/calculate_performance_metrics.py {predict_fp} {procap_fp} {out_fp}"
        print(cmd)
        os.system(cmd)
    else:
        print(f"{out_fp} exists, skipping.")
```

## Calculate individual-specific performance metrics

```python
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

out_dir = Path("/home2/ayh8/predictions/lcl_subsample/")
procap_fp = "/home2/ayh8/data/lcl/fixed_windows/concat_procap_0.csv.gz"

procap = pd.read_csv(procap_fp, index_col=0, header=None)

runs = range(5)
for r in runs:
    names = tuple(filter(None, Path(f"../clipnet_data_folds/subsample_prefixes_n1_run{r}.txt").read_text().split("\n")))
    out_fp = out_dir.joinpath(f"n1_run{r}_prediction.hdf5")
    with h5py.File(out_fp, "r") as hf:
        profile = hf["track"][:]
        quantity = hf["quantity"][:]
        prediction = (profile / np.sum(profile, axis=1)[:, None]) * quantity
    # filter rows by name
    mask = procap.index.str.startswith(names)
    procap_sub = np.array(procap)[mask][:, np.r_[250:750, 1250:1750]]
    prediction_sub = prediction[mask]
    # calculate performance metrics
    profile_pearson = pd.DataFrame(procap_sub).corrwith(pd.DataFrame(prediction_sub), axis=1)
    quantity_pcc = pearsonr(np.log(procap_sub.sum(axis=1) + 1e-3), np.log(prediction_sub.sum(axis=1) + 1e-3))[0]
    print(f"Run {r}: {profile_pearson.median()}, {quantity_pcc}")
```
