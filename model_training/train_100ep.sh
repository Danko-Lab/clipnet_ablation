#!/bin/bash -l
#SBATCH --job-name=ref100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tP100:2
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --array=1-9%1

# SLURM worker for train_100ep_queue.sh: each array task trains the next
# unclaimed fold (one ~38 h fold per task, within the 3-day gpu limit). Tasks
# that find every fold claimed exit immediately. Submit from model_training/ on
# the cbsubscb login node:
#
#   sbatch train_100ep.sh                                  # ref_model
#   sbatch --job-name=clip100 --export=ALL,ARM=clipnet train_100ep.sh

conda activate clipnet

# Checkpoints are written to cbsubscb17 lab storage (OUT_ROOT in the queue script).
/programs/bin/labutils/mount_server cbsubscb17 /storage

ONE_FOLD=1 bash train_100ep_queue.sh "${ARM:-ref_model}"
