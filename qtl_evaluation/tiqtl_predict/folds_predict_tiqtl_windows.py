import itertools
import json
import os
from pathlib import Path

file_name_hash = "../../data_spec/procap_to_1k_genomes.json"
with open(file_name_hash, "r") as handle:
    d = json.load(handle)

with open("../../data_spec/missing_1k_genomes.txt", "r") as handle:
    missing = handle.read().splitlines()


procap_prefixes = list(d.keys())
nonempty_procap_prefixes = [
    prefix for prefix in procap_prefixes if d[prefix] not in missing
]

folds = range(1, 10)

predict_dir = Path("../../predictions/tiqtl/fold_predict")
predict_dir.mkdir(exist_ok=True, parents=True)

clipnet_install = "/home2/ayh8/clipnet"
n_individuals = [5, 10, 15, 20, 30]
run = range(5)

for n, r, fold in itertools.product(n_individuals, run, folds):
    model_dir = Path(f"../../models/n{n}_run{r}/")
    model_fp = model_dir.joinpath(f"fold_{fold}.h5")
    outdir = Path(predict_dir, f"n{n}_run{r}/fold_{fold}")
    outdir.mkdir(exist_ok=True, parents=True)
    for prefix in nonempty_procap_prefixes:
        output = os.path.join(outdir, f"{prefix}.h5")
        if not os.path.exists(output):
            sequence = f"/home2/ayh8/data/lcl/tiqtl/sequence/{prefix}.fna.gz"
            cmd = f"python {clipnet_install}/predict_individual_model.py \
                    {model_fp} {sequence} {output} --gpu 0"
            os.system(f"echo {cmd}")
            os.system(cmd)
        else:
            print(f"{output} exists, skipping.")
