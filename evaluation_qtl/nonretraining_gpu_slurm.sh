#!/bin/bash -l
#SBATCH --job-name=qtl_bench
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tP100:2
#SBATCH --chdir=/home2/ayh8/clipnet_ablation/evaluation_qtl
#SBATCH --output=/home2/ayh8/clipnet_ablation/evaluation_qtl/logs/qtl_bench.%j.out

# cbsubscb worker pair for nonretraining_gpu_queue.sh: one queue worker per P100,
# sharing the claim list with any workers on cbsugpu01, so this job takes only
# the benchmark jobs nobody has started. Submit from a cbsubscb login node:
#
#   sbatch /home2/ayh8/clipnet_ablation/evaluation_qtl/nonretraining_gpu_slurm.sh
#
# A job cut off by the time limit leaves its claim without done/failed; remove
# that claim directory and resubmit (finished predictions are skipped).

set -euo pipefail

# Models and predictions live on cbsubscb17's lab storage.
/programs/bin/labutils/mount_server cbsubscb17 /storage

cd /home2/ayh8/clipnet_ablation/evaluation_qtl
nvidia-smi --query-gpu=index,name,memory.total --format=csv

bash nonretraining_gpu_queue.sh 0 &
sleep 30  # stagger so the workers claim different jobs cleanly
bash nonretraining_gpu_queue.sh 1 &
wait
