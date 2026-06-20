#!/bin/bash -l
#SBATCH --job-name=clipnet_ablation
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --array=9-26
#SBATCH -A bio240062p

set -eo pipefail
conda activate clipnet
set -u

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -n "${CLIPNET_ABLATION_ROOT:-}" ]]; then
    PROJECT_ROOT="${CLIPNET_ABLATION_ROOT}"
elif [[ -f "${SUBMIT_DIR}/model_training/fit.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/fit.py" && "$(basename "${SUBMIT_DIR}")" == "model_training" ]]; then
    PROJECT_ROOT="$(dirname "${SUBMIT_DIR}")"
else
    echo "Could not locate the clipnet_ablation repository from ${SUBMIT_DIR}." >&2
    echo "Submit from the repository root or model_training/, or set CLIPNET_ABLATION_ROOT." >&2
    exit 1
fi

TRAINING_DIR="${PROJECT_ROOT}/model_training"
MODEL_ROOT="${PROJECT_ROOT}/models"
RUNS=(clipnet mean_model ref_model)

task_id="${SLURM_ARRAY_TASK_ID}"
run_index=$((task_id / 9))
fold=$((task_id % 9 + 1))
run="${RUNS[$run_index]}"
fold_dir="${MODEL_ROOT}/${run}/f${fold}"

if [[ ! -f "${fold_dir}/dataset_params.json" ]]; then
    echo "Missing dataset parameters: ${fold_dir}/dataset_params.json" >&2
    exit 1
fi

echo "Training run=${run} fold=${fold}"
echo "Model directory: ${fold_dir}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"

time python "${TRAINING_DIR}/fit.py" \
    "${fold_dir}" \
    --name "fold_${fold}" \
    --n_gpus 2
