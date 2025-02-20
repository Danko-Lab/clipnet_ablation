# CLIPNET ablation

This repo contains scripts and notebooks that document ablation studies that we performed on [CLIPNET](https://www.biorxiv.org/content/10.1101/2024.03.13.583868). These studies were performed as part of our investigation into the extent to which training on matched genomic sequences and molecular profiles improves molecular QTL prediction, which we describe in [this preprint](https://www.biorxiv.org/content/10.1101/2024.10.15.618510).

The models & data generated by these analyses are deposited on [Zenodo](https://zenodo.org/records/14037356).

## Contents

`data_processing_scripts` contains many utility scripts used to process data for training the ablated CLIPNET models.

`data_spec` contains metadata info and config files that are used in some of the data processing scripts.

`snakemake.yml` describes the conda/environment used for the data processing pipelines. Note that this is separate from the dependencies needed for CLIPNET.

`clipnet_data_folds` contains snakemake pipelines to download and process data for the ablated model training.

`model_training` contains example scripts for training the ablated models.

`evaluation_across_loci` contains notebooks (& instructions) for plotting the accuracy of the ablated models at predicting PRO-cap signal across genomic loci.

`evaluation_qtl` contains notebooks (& instructions) for plotting the accuracy of the ablated models at predicting initiation QTL effects.

`example_predictions` contains a bunch of notebooks used for plotting predictions/attributions at selected example loci. These were not used in the paper and are rather preliminary.
