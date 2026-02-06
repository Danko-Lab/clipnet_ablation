#!/bin/bash -l
#SBATCH --job-name=diqtl_f
#SBATCH --time=72:00:00
#SBATCH --partition=long7,long30
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu
#SBATCH --array=0-6

# set up environment
conda activate clipnet

# run script:
n_individuals=(5 10 15 20 30 40 50)
i=$SLURM_ARRAY_TASK_ID
n=${n_individuals[$i]}
cat folds.sh | grep "n${n}_run" | sh