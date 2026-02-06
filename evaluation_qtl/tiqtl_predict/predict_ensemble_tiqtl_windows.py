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

predict_dir = Path("../../predictions/tiqtl/ensemble_predict")
predict_dir.mkdir(exist_ok=True, parents=True)

n_individuals = [5, 10, 15, 20, 30, 40, 50]
run = range(5)
for n, r in itertools.product(n_individuals, run):
    model_dir = Path(f"../../models/n{n}_run{r}/")
    outdir = Path(predict_dir, f"n{n}_run{r}")
    outdir.mkdir(exist_ok=True, parents=True)
    for prefix in nonempty_procap_prefixes:
        output = os.path.join(outdir, f"{prefix}.npz")
        if not os.path.exists(output):
            sequence = os.path.join("../../data/tiqtl/sequence", f"{prefix}.fna.gz")
            cmd = (
                f"clipnet predict -v -f {sequence} -o {output} -m {model_dir} --gpu -1"
            )
            print(cmd)
            # os.system(cmd)
            # time.sleep(1)
        else:
            # print(f"{output} exists, skipping.")
            pass
