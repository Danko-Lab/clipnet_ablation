#!/bin/bash

# GPU inference for the non-retraining clipnet vs ref_model analyses:
#   - main (early-stopped, best-val-loss) ensemble_models and reference_models,
#     scored with the current pipeline (composite = ensemble + held-out folds);
#   - every 5-epoch checkpoint of the 100-epoch five_epoch runs (clipnet via the
#     one-run-per-fold symlink view, ref_model as trained, with early stopping).
#
# Run one worker per GPU from evaluation_qtl/; workers share the job list
# through atomic mkdir claims, and every job uses --skip_existing, so a killed
# job can be retried by removing its claim:
#
#   screen -S qtl_gpu0 bash nonretraining_gpu_queue.sh 0
#   screen -S qtl_gpu1 bash nonretraining_gpu_queue.sh 1
#
# About 3.7 min per checkpoint for tiQTL on a TITAN Xp (67 individuals);
# ~35 GPU-hours in total.

set -euo pipefail

GPU="${1:?usage: nonretraining_gpu_queue.sh GPU_INDEX}"
PY="${PY:-${HOME}/miniconda3/envs/clipnet/bin/python}"
ABLATION="${ABLATION:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation}"
MAIN_ROOT="${MAIN_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${ABLATION}/revision_predictions}"
DATA_ROOT="${DATA_ROOT:-../data}"
CLAIMS="${PREDICTIONS_ROOT}/.nonretraining_claims"
# Pin the worker to its GPU; TensorFlow then sees it as device 0.
export CUDA_VISIBLE_DEVICES="${GPU}"

cd "$(dirname "$0")"
mkdir -p "${CLAIMS}"

common() {
    local qtl=$1
    echo --qtl "${qtl}" --predictions_root "${PREDICTIONS_ROOT}" \
        --data_root "${DATA_ROOT}" --qtl_data_dir "${DATA_ROOT}/${qtl}" \
        --experimental_l2_archive "${DATA_ROOT}/qtl_analysis.tar.gz" \
        --skip_existing --gpu --use_specific_gpu 0
}

# name|command, longest first so the two workers finish close together.
JOBS=()
for qtl in tiqtl diqtl; do
    JOBS+=("five_epoch_clipnet_${qtl}|${PY} benchmark_epoch_checkpoints.py all ${ABLATION}/models/five_epoch_clipnet_clean --mode folds --run_name five_epoch_clipnet $(common ${qtl})")
done
for qtl in tiqtl diqtl; do
    JOBS+=("five_epoch_ref_model_${qtl}|${PY} benchmark_epoch_checkpoints.py all ${ABLATION}/models/five_epoch/ref_model --mode folds --allow_missing_folds --run_name five_epoch_ref_model $(common ${qtl})")
    for models in ensemble_models reference_models; do
        JOBS+=("main_${models}_${qtl}|${PY} benchmark_published_clipnet.py all ${MAIN_ROOT}/${models} --mode composite --run_name main_${models} $(common ${qtl})")
    done
done

for job in "${JOBS[@]}"; do
    name="${job%%|*}"
    cmd="${job#*|}"
    mkdir "${CLAIMS}/${name}" 2>/dev/null || continue
    echo "[gpu${GPU} $(date -Is)] start ${name}"
    log="${CLAIMS}/${name}/log.txt"
    if { time ${cmd} ; } >"${log}" 2>&1; then
        touch "${CLAIMS}/${name}/done"
        echo "[gpu${GPU} $(date -Is)] done ${name}"
    else
        echo "[gpu${GPU} $(date -Is)] FAILED ${name}; see ${log}"
    fi
done
echo "[gpu${GPU}] no unclaimed jobs left"
