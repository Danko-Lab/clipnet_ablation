#!/bin/bash -l
#SBATCH --job-name=n40_run2
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:l48s-40:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --array=1-9
#SBATCH -A bio240062p

# set up environment
conda activate clipnet

# run script:
i=$SLURM_ARRAY_TASK_ID
time python fit.py ../models/n40_run2/f${i}/ --name fold_${i}