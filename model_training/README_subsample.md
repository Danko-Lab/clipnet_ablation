# Subsampled CLIPNET

Here we'll cover training of the subsampled CLIPNET models. For this ablation test, we subsampled the 58 genetically distinct LCL PRO-cap datasets down to n=5,10,15,20,30 individuals, then trained CLIPNET models on the subsampled dataset. The scripts to do this subsampling are in `../clipnet_data_folds`. Once the subsample data have been generated, the subsampled models can be trained as follows.

We use the training scripts in the original [CLIPNET repo](https://github.com/Danko-Lab/clipnet/). So please first clone this repo & install the dependencies as described there. Then, we'll first use `calculate_dataset_params.py` to generate jsons with the model hyperparameters (steps per epoch, input data file paths, etc.):

```bash
cd clipnet_path/
python calculate_dataset_params.py data_path/ model_path/

# where clipnet_path is the path to the CLIPNET repo path,
# data_path/ is the path to the data for a particular subsampled dataset
# and model_path/ is the path to where the model hyperparameters will be written to
```

The `_fit_nn.py` script can then be used to train the models. The `fit_nn.py` script will automatically save the model weights and training progress to the output directory, so you can check on the training progress at any time. It takes as input the `f*` directories created by the `calculate_dataset_params.py` script:

```bash
for fold in {1..9}; do
    python _fit_nn.py model_path/f${fold} --gpu 0;
done
# where model_path/ is as above. --gpu 0 will train on the first GPU.
```

This will run a for loop to train the models on each of the 9 folds for a given subsampled dataset.

The models that we trained (as well as the individual IDs of each subsampling expt.) are on [Zenodo](https://zenodo.org/records/14037356).