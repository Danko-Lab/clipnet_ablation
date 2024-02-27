# Evaluation of subsampled models

## Prediction commands

```python
import os
from pathlib import Path

clipnet_install = "/home2/ayh8/clipnet"
out_dir = Path("/home2/ayh8/predictions/subsample/")
out_dir.mkdir(exist_ok=True, parents=True)
fasta_fp = Path("/home2/ayh8/data/gse110638/fixed_windows/data_folds/sequence/").joinpath("concat_sequence_0.fna.gz")

for run in ["n5_run0", "n10_run0", "n15_run0", "n20_run0"]:
    model_dir = Path("/home2/ayh8/subsample_models/").joinpath(run)
    out_fp = out_dir.joinpath(f"ensemble_{run}_fold_0_predictions.h5")
    cmd = f"python {clipnet_install}/predict_ensemble.py {fasta_fp} {out_fp} --model_dir {str(model_dir)} --gpu --low_mem"
    print(cmd)
    os.system(cmd)
```
