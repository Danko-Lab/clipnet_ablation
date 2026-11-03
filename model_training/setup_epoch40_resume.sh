#!/bin/bash

# Create replicate fold directories for the epoch-40 -> epoch-45 resume experiment.
#
#   OUT_ROOT/{clipnet,ref_model}/rep{01..20}/f{1..9}/dataset_params.json
#
# Every replicate resumes from the same BASE_ROOT/{arm}/f{fold}/clipnet_epoch_040.hdf5
# (the 50-epoch, no-early-stopping, every-5-epoch-checkpoint runs). Run from
# model_training/ so relative data paths in dataset_params.json resolve.

set -euo pipefail

BASE_ROOT="${BASE_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/models/five_epoch/_old}"
OUT_ROOT="${OUT_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/models/epoch40_resume}"
N_REPS="${N_REPS:-20}"
ARMS=(clipnet ref_model)

for arm in "${ARMS[@]}"; do
    for fold in {1..9}; do
        base="${BASE_ROOT}/${arm}/f${fold}"
        for file in dataset_params.json clipnet_epoch_040.hdf5; do
            if [[ ! -f "${base}/${file}" ]]; then
                echo "Missing ${base}/${file}" >&2
                exit 1
            fi
        done
        for rep in $(seq -f "%02g" 1 "${N_REPS}"); do
            out="${OUT_ROOT}/${arm}/rep${rep}/f${fold}"
            mkdir -p "${out}"
            cp "${base}/dataset_params.json" "${out}/"
        done
    done
done

echo "Created ${#ARMS[@]} arms x ${N_REPS} replicates x 9 folds under ${OUT_ROOT}"
