#!/bin/bash

# Train ablation folds to 100 epochs with no early stopping, retaining a
# checkpoint every 5 epochs, on 2 GPUs (batch 512, lr 0.002). This matches the
# existing five_epoch/clipnet runs.
#
# Several workers (this server, SLURM jobs) can run this at once: each claims
# the next unclaimed fold with an atomic mkdir under OUT_ROOT/.claims, so no fold
# is trained twice. A worker keeps taking folds until none remain, or trains
# one fold and exits when ONE_FOLD=1 (used by the SLURM array).
#
#   bash train_100ep_queue.sh ref_model          # from model_training/
#   bash train_100ep_queue.sh clipnet            # optional clipnet rerun
#
# A fold whose job dies keeps its claim; remove OUT_ROOT/.claims/f<fold> to retry.

set -euo pipefail

ARM="${1:?usage: train_100ep_queue.sh ARM (ref_model, clipnet, or mean_model)}"
PARAMS_ROOT="${PARAMS_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/models/five_epoch}"
OUT_ROOT="${OUT_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/models/${ARM}_100ep}"
ONE_FOLD="${ONE_FOLD:-0}"
# Call the env interpreter directly: a virtualenv on PATH can shadow conda's python.
PY="${PY:-${HOME}/miniconda3/envs/clipnet/bin/python}"
WORKER="$(hostname -s)${SLURM_JOB_ID:+_slurm${SLURM_JOB_ID}}"

cd "$(dirname "$0")"
mkdir -p "${OUT_ROOT}/.claims"

for fold in {1..9}; do
    claim="${OUT_ROOT}/.claims/f${fold}"
    mkdir "${claim}" 2>/dev/null || continue
    echo "${WORKER} $(date -Is)" > "${claim}/worker"

    model_dir="${OUT_ROOT}/f${fold}"
    mkdir -p "${model_dir}"
    cp "${PARAMS_ROOT}/${ARM}/f${fold}/dataset_params.json" "${model_dir}/"

    echo "[${WORKER}] ${ARM} fold ${fold} -> ${model_dir} ($(date -Is))"
    nvidia-smi --query-gpu=name --format=csv,noheader
    { time "${PY}" fit.py "${model_dir}" \
        --name "fold_${fold}" \
        --epochs 100 \
        --checkpoint_frequency 5 \
        --seed "${fold}" \
        --n_gpus 2 ; } 2>&1 | tee "${model_dir}/train_${WORKER}.out"
    echo "${WORKER} $(date -Is)" > "${claim}/done"

    [[ "${ONE_FOLD}" == 1 ]] && break
done
echo "[${WORKER}] no unclaimed ${ARM} folds left"
