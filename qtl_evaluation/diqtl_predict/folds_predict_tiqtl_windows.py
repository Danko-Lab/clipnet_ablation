import json
import os
from pathlib import Path

with open("../data_spec/lcl_proc_spec.json", "r") as f:
    dirs = json.load(f)
    DATADIR = dirs["DATADIR"]
    PREDICTDIR = dirs["PREDICTDIR"]


file_name_hash = "../data_spec/procap_to_1k_genomes.json"
with open(file_name_hash, "r") as handle:
    d = json.load(handle)

with open("../data_spec/missing_1k_genomes.txt", "r") as handle:
    missing = handle.read().splitlines()


procap_prefixes = list(d.keys())
nonempty_procap_prefixes = [
    prefix for prefix in procap_prefixes if d[prefix] not in missing
]

folds = range(1, 10)

predict_dir = Path("/home2/ayh8/predictions/ensemble/tiqtl/individual_folds/")
predict_dir.mkdir(exist_ok=True, parents=True)

for fold in folds:
    model_dir = Path("/home2/ayh8/ensemble_models/")
    model_fp = model_dir.joinpath(f"fold_{fold}.h5")
    os.makedirs(os.path.join(predict_dir, f"tiqtl_{fold}"), exist_ok=True)
    for prefix in nonempty_procap_prefixes:
        output = os.path.join(predict_dir, f"tiqtl_{fold}", "%s.h5" % prefix)
        if not os.path.exists(output):
            sequence = os.path.join(
                "/home2/ayh8/data/gse110638", "tiqtl/sequence", f"{prefix}.fna.gz"
            )
            procap = os.path.join(
                "/home2/ayh8/data/gse110638", "tiqtl/procap", f"{prefix}.csv.gz"
            )
            cmd = f"python /home2/ayh8/clipnet_scripts/clipnet/predict_on_fasta.py \
                    {sequence} {output} --model_fp {model_fp} --n_gpus 1"
            os.system(f"echo {cmd}")
            os.system(cmd)
        else:
            print(f"{output} exists, skipping.")
