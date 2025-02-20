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

predict_dir = Path("/mnt/i/clipnet_data/tiqtl/ensemble_predict")
predict_dir.mkdir(exist_ok=True, parents=True)

clipnet_install = "~/github/clipnet"
n_individuals = [5, 10, 15, 20, 30]
run = range(5)
for n, r in itertools.product(n_individuals, run):
    model_dir = Path(f"../../models/n{n}_run{r}/")
    outdir = Path(predict_dir, f"n{n}_run{r}")
    outdir.mkdir(exist_ok=True, parents=True)
    for prefix in nonempty_procap_prefixes:
        output = os.path.join(outdir, f"{prefix}.h5")
        if not os.path.exists(output):
            sequence = os.path.join(
                "../../../clipnet_data/tiqtl/sequence", f"{prefix}.fna.gz"
            )
            cmd = f"clipnet predict -f {sequence} -o {output} -m {model_dir}"
            os.system(f"echo {cmd}")
            os.system(cmd)
        else:
            print(f"{output} exists, skipping.")
