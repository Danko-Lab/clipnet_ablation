# Evaluation of subsampled models

## Prediction commands

```python
import os
from pathlib import Path

out_dir = Path("/home2/ayh8/predictions/subsample/")
out_dir.mkdir(exist_ok=True, parents=True)
model_dir = Path("/home2/ayh8/ensemble_models/")
fasta_fp = Path("/home2/ayh8/data/gse110638/fixed_windows/data_folds/sequence/").joinpath("concat_sequence_0.fna.gz")
out_fp = out_dir.joinpath(f"ensemble_fold_0_predictions.h5")
cmd = f"python predict_ensemble.py {fasta_fp} {out_fp} --model_dir {str(model_dir)} --n_gpus 1 --low_mem --use_specific_gpu 0 --reverse_complement"

print(cmd)
os.system(cmd)
```
