#!/bin/sh
#SBATCH --job-name=clipnet_30_2
#SBATCH --time=48:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mail-type=all
#SBATCH --mail-user=ayh8@cornell.edu

module load anaconda3
conda activate tf
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/jet/home/adamyhe/.conda/envs/tf/lib/

n=30
run=2

scratch=$LOCAL/adamyhe/$SLURM_JOB_ID
mkdir -p $scratch
cd $scratch

# Copy the necessary files to the scratch directory
cp -r ~/storage/adamyhe/clipnet .
cp -r ~/storage/adamyhe/clipnet_subsampling/data/${n}_subsample_run${run} .
cp /jet/home/adamyhe/.conda/envs/tf/lib/libdevice.10.bc clipnet/

# Calculate parameters for this data fold
cd clipnet/
python calculate_dataset_params.py \
    ../${n}_subsample_run${run} \
    ~/storage/adamyhe/clipnet_subsampling/models/n${n}_run${run}/ \
    --threads 1

# Train the model
for fold in {1..9}; do
    python fit_nn.py ~/storage/adamyhe/clipnet_subsampling/models/n${n}_run${run}/f${fold} --gpu 0;
done

# Cleanup
cd $LOCAL
rm -r $scratch