#!/bin/bash -l
#SBATCH --job-name=mean
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --array=2-9

conda activate clipnet

i=$SLURM_ARRAY_TASK_ID
time python fit.py ../models/mean_model/f${i}
