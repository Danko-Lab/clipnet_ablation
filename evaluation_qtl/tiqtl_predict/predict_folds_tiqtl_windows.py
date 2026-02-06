import itertools
import json
import os
import time  # noqa
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

n_individuals = [5, 10, 15, 20, 30, 40, 50]
run = range(5)
for n, r, fold in itertools.product(n_individuals, run, folds):
    model_dir = Path(f"../../models/n{n}_run{r}/")
    model_fp = model_dir.joinpath(f"fold_{fold}.h5")
    if not os.path.exists(model_fp):
        model_fp = model_dir.joinpath(f"fold_{fold}.hdf5")
        if not os.path.exists(model_fp):
            raise ValueError(f"Model {model_fp} does not exist.")
    outdir = Path(predict_dir, f"n{n}_run{r}/fold_{fold}")
    outdir.mkdir(exist_ok=True, parents=True)
    for prefix in nonempty_procap_prefixes:
        output = os.path.join(outdir, f"{prefix}.npz")
        if not os.path.exists(output):
            sequence = f"../../data/tiqtl/sequence/{prefix}.fna.gz"
            cmd = f"clipnet predict -v -f {sequence} -o {output} -m {model_fp} --gpu -1"
            print(cmd)
            # os.system(cmd)
            # time.sleep(1)
        else:
            # print(f"{output} exists, skipping.")
            pass
