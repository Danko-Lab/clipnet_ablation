#!/bin/sh
#SBATCH --job-name=fit_fold1
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu

module load anaconda3
conda activate tf
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/jet/home/adamyhe/.conda/envs/tf/lib/

n=$1
run=$2
fold=$3

scratch=$LOCAL/adamyhe/$SLURM_JOB_ID
mkdir -p $scratch
cd $scratch
cp -r ~/storage/adamyhe/clipnet .
cp -r ~/storage/adamyhe/final_data_folds .
cp ~/storage/adamyhe/clipnet_old/calculate_fold_params_psc.py clipnet/

cd $LOCAL/adamyhe/clipnet
cp /jet/home/adamyhe/.conda/envs/tf/lib/libdevice.10.bc .

# calculate parameters for this data fold
python calculate_fold_params_psc.py $LOCAL/adamyhe/final_data_folds $fold

# train the model
python fit_nn.py /ocean/projects/bio210011p/adamyhe/dilated_models/f$(($fold+1)) \
	--n_gpus 1 --prefix rnn_v11

rm -r $LOCAL/adamyhe/