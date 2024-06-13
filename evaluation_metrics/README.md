# Evaluation of subsampled models

## Prediction commands

```python
import os
import itertools
from pathlib import Path

clipnet_install = "/home2/ayh8/clipnet"
out_dir = Path("/home2/ayh8/clipnet_subsampling/predictions/overall")
out_dir.mkdir(exist_ok=True, parents=True)
fasta_fp = Path("/home2/ayh8/data/lcl/fixed_windows/").joinpath("concat_sequence_0.fna.gz")
procap_fp = Path("/home2/ayh8/data/lcl/fixed_windows/").joinpath("concat_procap_0.npz")

n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

for n, r in itertools.product(n_individuals, runs):
    run = f"n{n}_run{r}"
    model_dir = Path("/home2/ayh8/clipnet_subsampling/models/").joinpath(run)
    out_fp = out_dir.joinpath(f"ensemble_{run}_fold_0_predictions.h5")
    if not os.path.exists(out_fp):
        cmd = f"python {clipnet_install}/predict_ensemble.py {fasta_fp} {out_fp} --model_dir {str(model_dir)} --gpu 1"
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
out_dir = Path("/home2/ayh8/clipnet_subsampling/predictions/overall")
out_dir.mkdir(exist_ok=True, parents=True)
procap_fp = Path("/home2/ayh8/data/lcl/fixed_windows/").joinpath("concat_procap_0.csv.gz")

n_individuals = [5, 10, 15, 20, 30]
runs = range(5)

for n, r in itertools.product(n_individuals, runs):
    run = f"n{n}_run{r}"
    predict_fp = out_dir.joinpath(f"ensemble_{run}_fold_0_predictions.h5")
    out_fp = out_dir.joinpath(f"ensemble_{run}_fold_0_performance_metrics.h5")
    if not os.path.exists(out_fp):
        cmd = f"python {clipnet_install}/calculate_performance_metrics.py {predict_fp} {procap_fp} {out_fp}"
        print(cmd)
        os.system(cmd)
    else:
        print(f"{out_fp} exists, skipping.")
```
