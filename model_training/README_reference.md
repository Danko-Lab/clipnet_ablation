# Variant masked/reference-only CLIPNET

Here we'll cover training of variant-masked or reference-only CLIPNET models. For this ablation test, we trained a model on the full 58 (+9 replicates) genetically distinct LCL PRO-cap libraries, but we masked out the variants from the personal genomes. That is, we retrained CLIPNET, but only using reference genome sequences. The scripts for extracting the PRO-cap peak signals & reference genome sequencces are in `../clipnet_data_folds` 
Once these data have been processed, the reference-only models can be trained as follows.

We use the training scripts in the original [CLIPNET repo](https://github.com/Danko-Lab/clipnet/). So please first clone this repo & install the dependencies as described there. Then, we'll first use `calculate_dataset_params.py` to generate jsons with the model hyperparameters (steps per epoch, input data file paths, etc.):

```bash
cd clipnet_path/
python calculate_dataset_params.py reference_data_path/ reference_model_path/

# where clipnet_path is the path to the CLIPNET repo path,
# reference_data_path/ is the path to the reference sequences + PRO-cap
# and reference_model_path/ is the path to where the model hyperparameters will be written to
```

The `_fit_nn.py` script can then be used to train the models. The `_fit_nn.py` script will automatically save the model weights and training progress to the output directory, so you can check on the training progress at any time. It takes as input the `f*` directories created by the `calculate_dataset_params.py` script:

```bash
for fold in {1..9}; do
    python _fit_nn.py reference_model_path/f${fold} --gpu 0;
done
# where reference_model_path/ is as above. --gpu 0 will train on the first GPU.
```

The model that we trained is deposited on Zenodo:

```bash
wget https://zenodo.org/records/14037356/files/reference_models.tar
tar -xvf reference_models.tar
```
