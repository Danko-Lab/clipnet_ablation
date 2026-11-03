#!/bin/bash -l
#SBATCH --job-name=e40resume
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --array=0-359%16

# Epoch-40 -> epoch-45 resume experiment: 2 arms x 20 replicates x 9 folds.
# Task index -> arm (0-179 clipnet, 180-359 ref_model), replicate, fold.
# Replicate r uses the same seed in both arms (both arms have identical sample
# counts, so batch order is matched), giving paired replicates.
# The base checkpoints were trained on 1 GPU (batch 256, lr 0.001; 17,230
# steps/epoch), so resume on 1 GPU to continue the same optimization regime.
# Run setup_epoch40_resume.sh first, and submit from model_training/.

conda activate clipnet

BASE_ROOT="${BASE_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/models/five_epoch/_old}"
OUT_ROOT="${OUT_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/models/epoch40_resume}"
ARMS=(clipnet ref_model)

i=$SLURM_ARRAY_TASK_ID
arm=${ARMS[$((i / 180))]}
rep=$(((i % 180) / 9 + 1))
fold=$((i % 9 + 1))
rep_id=$(printf "%02d" "${rep}")
model_dir="${OUT_ROOT}/${arm}/rep${rep_id}/f${fold}"

if [[ -f "${model_dir}/clipnet_resume_epoch_045.hdf5" ]]; then
    echo "Already finished: ${model_dir}"
    exit 0
fi
# A partial run would append to the CSV log; start each attempt clean.
rm -f "${model_dir}"/clipnet_resume_* "${model_dir}/clipnet.log" "${model_dir}/clipnet_history.json"

echo "arm=${arm} rep=${rep_id} fold=${fold} seed=$((1000 * rep + fold))"
time python fit.py "${model_dir}" \
    --name clipnet \
    --resume_checkpoint "${BASE_ROOT}/${arm}/f${fold}/clipnet_epoch_040.hdf5" \
    --restore_optimizer \
    --initial_epoch 40 \
    --epochs 5 \
    --seed $((1000 * rep + fold)) \
    --n_gpus 1
