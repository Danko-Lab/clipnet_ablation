# Extracting example windows

## NOT USED IN PERSONALIZED GENOMES PAPER

This directory contains scripts for extracting sample windows.

```bash
cat lcl_cre_example_windows.bed qtl_snps_windows.bed | bedtools sort > example_windows.bed
```

Run the snakemake pipelines in `procap`, and `sequence` to extract data for each of the windows in `example_windows.bed`.

Then, run the prediction using:

```bash
cd /home2/ayh8/clipnet/
conda activate clipnet
```

Individual models:

```python
import itertools
import os

n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

for n, r in itertools.product(n_individuals, runs):
    run = f"n{n}_run{r}"
    for f in range(1, 10):
        cmd = f"python predict_individual_model.py \
            ../clipnet_subsampling/models/{run}/fold_{f}.h5 \
            ../data/lcl/examples/concat_sequence.fna.gz \
            ../predictions/lcl_subsample/examples/{run}_fold_{f}_examples_prediction.h5 \
            --gpu 0"
        print(cmd)
        os.system(cmd)
```

Ensemble models:

```python
import itertools
import os

n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

for n, r in itertools.product(n_individuals, runs):
    run = f"n{n}_run{r}"
    cmd = f"python predict_ensemble.py \
        ../data/lcl/examples/concat_sequence.fna.gz \
        ../predictions/lcl_subsample/examples/{run}_ensemble_examples_prediction.h5 \
        --model_dir ../clipnet_subsampling/models/{run} \
        --gpu 0"
    print(cmd)
    os.system(cmd)
```

Calculate DeepSHAP scores from model ensembles:

```python
import itertools
import os

n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

for n, r in itertools.product(n_individuals, runs):
    run = f"n{n}_run{r}"
    cmd = f"python calculate_deepshap.py \
        ../data/lcl/examples/concat_sequence.fna.gz \
        ../data/attribution_scores/examples_deepshap_{run}.npz \
        ../data/test_onehot.npz \
        --mode quantity \
        --gpu"
    print(cmd)
    os.system(cmd)
```

```bash
python /home2/ayh8/clipnet/calculate_deepshap.py \
    /home2/ayh8/data/lcl/examples/concat_sequence.fna.gz \
    /home2/ayh8/data/attribution_scores/examples_deepshap_quantity.npz \
    data/test_onehot.npz \
    --mode quantity \
    --gpu
```
