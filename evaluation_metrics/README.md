# Evaluation of subsampled models

## Prediction commands

```python
import os
from pathlib import Path

clipnet_install = "/home2/ayh8/clipnet"
out_dir = Path("/home2/ayh8/clipnet_subsampling/predictions/")
out_dir.mkdir(exist_ok=True, parents=True)
fasta_fp = Path("/home2/ayh8/data/lcl/fixed_windows/").joinpath("concat_sequence_0.fna.gz")
procap_fp = Path("/home2/ayh8/data/lcl/fixed_windows/").joinpath("concat_procap_0.npz")

runs = ["n5_run0", "n5_run1", "n10_run0", "n10_run1", "n15_run0", "n15_run1", "n20_run0", "n20_run1", "n20_run2", "n20_run3", "n20_run4", "n30_run0", "n30_run1"]

for run in runs:
    model_dir = Path("/home2/ayh8/clipnet_subsampling/models/").joinpath(run)
    out_fp = out_dir.joinpath(f"ensemble_{run}_fold_0_predictions.h5")
    cmd = f"python {clipnet_install}/predict_ensemble.py {fasta_fp} {out_fp} --model_dir {str(model_dir)} --gpu 0"
    print(cmd)
    os.system(cmd)

for run in runs:
    model_dir = Path("/home2/ayh8/clipnet_subsampling/models/").joinpath(run)
    for fold in range(1, 10):
        out_fp = out_dir.joinpath(f"fold_{fold}_{run}_fold_0_predictions.h5")
        cmd = f"python {clipnet_install}/calculate_performance_metrics.py {model_dir.joinpath(f'f{fold}.h5')} {procap_fp} {out_fp}"
        print(cmd)
        os.system(cmd)
```

## Predict on tiQTLs

```python
import json
import os
from pathlib import Path

clipnet_install = Path("/home2/ayh8/clipnet")

file_name_hash = "../data_spec/procap_to_1k_genomes.json"
with open(file_name_hash, "r") as handle:
    d = json.load(handle)

with open("../data_spec/missing_1k_genomes.txt", "r") as handle:
    missing = handle.read().splitlines()

procap_prefixes = list(d.keys())
nonempty_procap_prefixes = [
    prefix for prefix in procap_prefixes if d[prefix] not in missing
]

for run in ["n5_run0", "n10_run0", "n15_run0", "n20_run0", "n30_run0"]:
    model_dir = Path("/home2/ayh8/subsample_models/").joinpath(run)
    predict_dir = Path("/home2/ayh8/predictions/subsample/tiqtl/ensemble_predictions/").joinpath(run)
    predict_dir.mkdir(exist_ok=True, parents=True)
    for prefix in nonempty_procap_prefixes:
        output = predict_dir.joinpath(f"{prefix}_tiQTL_predictions.h5")
        if not os.path.exists(output):
            sequence = os.path.join(
                "/home2/ayh8/data/gse110638", "tiqtl/sequence", f"{prefix}.fna.gz"
            )
            procap = os.path.join(
                "/home2/ayh8/data/gse110638", "tiqtl/procap", f"{prefix}.csv.gz"
            )
            cmd = f"python {str(clipnet_install)}/predict_ensemble.py \
                    {str(sequence)} {str(output)} \
                    --model_dir {str(model_dir)} --gpu 0"
            os.system(f"echo {cmd}")
            os.system(cmd)
        else:
            print(f"{output} exists, skipping.")
```

## Predict on diQTLs

```python
import json
import os
from pathlib import Path

clipnet_install = Path("/home2/ayh8/clipnet")

file_name_hash = "../data_spec/procap_to_1k_genomes.json"
with open(file_name_hash, "r") as handle:
    d = json.load(handle)

with open("../data_spec/missing_1k_genomes.txt", "r") as handle:
    missing = handle.read().splitlines()

procap_prefixes = list(d.keys())
nonempty_procap_prefixes = [
    prefix for prefix in procap_prefixes if d[prefix] not in missing
]

for run in ["n5_run0", "n10_run0", "n15_run0", "n20_run0", "n30_run0"]:
    model_dir = Path("/home2/ayh8/subsample_models/").joinpath(run)
    predict_dir = Path("/home2/ayh8/predictions/subsample/diqtl/ensemble_predictions/").joinpath(run)
    predict_dir.mkdir(exist_ok=True, parents=True)
    for prefix in nonempty_procap_prefixes:
        output = predict_dir.joinpath(f"{prefix}_diQTL_predictions.h5")
        if not os.path.exists(output):
            sequence = os.path.join(
                "/home2/ayh8/data/gse110638", "diqtl/sequence", f"{prefix}.fna.gz"
            )
            procap = os.path.join(
                "/home2/ayh8/data/gse110638", "diqtl/procap", f"{prefix}.csv.gz"
            )
            cmd = f"python {str(clipnet_install)}/predict_ensemble.py \
                    {str(sequence)} {str(output)} \
                    --model_dir {str(model_dir)} --gpu --low_mem --use_specific_gpu 1"
            os.system(f"echo {cmd}")
            os.system(cmd)
        else:
            print(f"{output} exists, skipping.")
```

## Predict on tiQTLs with individual models

```python
import json
import os
from pathlib import Path

clipnet_install = Path("/home2/ayh8/clipnet")

file_name_hash = "../data_spec/procap_to_1k_genomes.json"
with open(file_name_hash, "r") as handle:
    d = json.load(handle)

with open("../data_spec/missing_1k_genomes.txt", "r") as handle:
    missing = handle.read().splitlines()

procap_prefixes = list(d.keys())
nonempty_procap_prefixes = [
    prefix for prefix in procap_prefixes if d[prefix] not in missing
]

for run in ["n5_run0", "n10_run0", "n15_run0", "n20_run0", "n30_run0"]:
    model_dir = Path("/home2/ayh8/subsample_models/").joinpath(run)
    for fold in range(1, 10):
        model_fp = model_dir.joinpath(f"fold_{fold}.h5")
        predict_dir = Path("/home2/ayh8/predictions/subsample/tiqtl/fold_predictions/").joinpath(run).joinpath(f"fold_{fold}")
        predict_dir.mkdir(exist_ok=True, parents=True)
        for prefix in nonempty_procap_prefixes:
            output = predict_dir.joinpath(f"{prefix}_tiQTL_predictions.h5")
            if not os.path.exists(output):
                sequence = os.path.join(
                    "/home2/ayh8/data/gse110638", "tiqtl/sequence", f"{prefix}.fna.gz"
                )
                procap = os.path.join(
                    "/home2/ayh8/data/gse110638", "tiqtl/procap", f"{prefix}.csv.gz"
                )
                cmd = f"python {str(clipnet_install)}/predict_individual_model.py \
                        {str(model_fp)} {str(sequence)} {str(output)} \
                        --gpu --low_mem"
                os.system(f"echo {cmd}")
                os.system(cmd)
            else:
                print(f"{output} exists, skipping.")
```

## Predict on diQTLs with individual models

```python
import json
import os
from pathlib import Path

clipnet_install = Path("/home2/ayh8/clipnet")

file_name_hash = "../data_spec/procap_to_1k_genomes.json"
with open(file_name_hash, "r") as handle:
    d = json.load(handle)

with open("../data_spec/missing_1k_genomes.txt", "r") as handle:
    missing = handle.read().splitlines()

procap_prefixes = list(d.keys())
nonempty_procap_prefixes = [
    prefix for prefix in procap_prefixes if d[prefix] not in missing
]

for run in ["n5_run0", "n10_run0", "n15_run0", "n20_run0", "n30_run0"]:
    model_dir = Path("/home2/ayh8/subsample_models/").joinpath(run)
    for fold in range(1, 10):
        model_fp = model_dir.joinpath(f"fold_{fold}.h5")
        predict_dir = Path("/home2/ayh8/predictions/subsample/diqtl/fold_predictions/").joinpath(run).joinpath(f"fold_{fold}")
        predict_dir.mkdir(exist_ok=True, parents=True)
        for prefix in nonempty_procap_prefixes:
            output = predict_dir.joinpath(f"{prefix}_diQTL_predictions.h5")
            if not os.path.exists(output):
                sequence = os.path.join(
                    "/home2/ayh8/data/gse110638", "diqtl/sequence", f"{prefix}.fna.gz"
                )
                procap = os.path.join(
                    "/home2/ayh8/data/gse110638", "diqtl/procap", f"{prefix}.csv.gz"
                )
                cmd = f"python {str(clipnet_install)}/predict_individual_model.py \
                        {str(model_fp)} {str(sequence)} {str(output)} \
                        --gpu --use_specific_gpu 1 --low_mem"
                os.system(f"echo {cmd}")
                os.system(cmd)
            else:
                print(f"{output} exists, skipping.")
```
