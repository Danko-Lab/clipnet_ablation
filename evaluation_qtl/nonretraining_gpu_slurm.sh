#!/bin/bash -l
#SBATCH --job-name=qtl_bench
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tP100:1
#SBATCH --array=1-6%2
#SBATCH --chdir=/home2/ayh8/clipnet_ablation/evaluation_qtl
#SBATCH --output=/home2/ayh8/clipnet_ablation/evaluation_qtl/logs/qtl_bench.%A_%a.out

# cbsubscb array for nonretraining_gpu_queue.sh: each task takes one P100 and
# runs the next unclaimed benchmark job, sharing the claim list with any
# workers on cbsugpu01. Tasks that find every job claimed exit immediately.
# Six tasks cover the jobs the cbsugpu01 workers had not started; %2 matches
# the node's two GPUs. Submit from a cbsubscb login node:
#
#   sbatch /home2/ayh8/clipnet_ablation/evaluation_qtl/nonretraining_gpu_slurm.sh
#
# A task cut off by the time limit leaves its claim without done/failed; remove
# that claim directory and resubmit one task (finished predictions are skipped):
#
#   sbatch --array=1 /home2/ayh8/clipnet_ablation/evaluation_qtl/nonretraining_gpu_slurm.sh

set -euo pipefail

# Models and predictions live on cbsubscb17's lab storage.
/programs/bin/labutils/mount_server cbsubscb17 /storage

cd /home2/ayh8/clipnet_ablation/evaluation_qtl
nvidia-smi --query-gpu=index,name,memory.total --format=csv

ONE_JOB=1 bash nonretraining_gpu_queue.sh 0
