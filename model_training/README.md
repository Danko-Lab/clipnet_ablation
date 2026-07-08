# CLIPNET ablation model training

This directory contains the current training entrypoint, architecture settings,
and dataset-parameter generators for the CLIPNET ablation models.

The default configuration is currently set up for short diagnostic retraining
runs with every-epoch retained checkpoints:

| Setting | Value |
|---|---:|
| GPUs | 2 |
| Per-GPU base batch size | 256 |
| Effective batch size | 512 |
| Per-GPU base learning rate | 0.001 |
| Effective learning rate | 0.002 |
| Maximum epochs | 20 |
| Early-stopping patience | 100 epochs |
| Best-checkpoint evaluation | Every epoch |
| Retained checkpoint frequency | Every epoch |

`clipnet_all_checkpoints.py` scales batch size and learning rate linearly with
the requested GPU count. The effective configuration is printed when training
starts and should be retained in the job log.

## Prepare fold directories

The training data directory must contain the processed sequence and PRO-cap
arrays for folds 1 through 9. Choose the parameter generator matching the
ablation:

```bash
# Personalized-genome CLIPNET
python model_training/calculate_fold_params.py DATA_DIR MODEL_ROOT

# Reference-sequence ablation
python model_training/calculate_fold_params_ref.py DATA_DIR MODEL_ROOT

# Mean-PRO-cap ablation
python model_training/calculate_fold_params_mean.py DATA_DIR MODEL_ROOT
```

Each command creates `MODEL_ROOT/f1` through `MODEL_ROOT/f9`, with a
`dataset_params.json` in every directory. For model fold `i`:

- Fold `i` is the test fold.
- Fold `(i % 9) + 1` is the validation fold.
- The remaining seven folds are used for training.

The parameter files contain absolute or supplied paths to the processed arrays,
sample counts, input/output lengths, padding, and the quantity-loss weight.
Verify these paths from the training host before launching jobs.

## Train models

Train one model fold with the default two-GPU configuration:

```bash
python model_training/fit.py MODEL_ROOT/f1 --name fold_1
```

Train all nine folds sequentially:

```bash
for fold in {1..9}; do
    python model_training/fit.py \
        MODEL_ROOT/f${fold} \
        --name fold_${fold}
done
```

The command requires two GPUs to be visible. It fails clearly instead of
silently changing the optimization regime when fewer GPUs are available.

Alternative GPU counts may be requested explicitly:

```bash
# One GPU: effective batch size 256 and learning rate 0.001
python model_training/fit.py MODEL_ROOT/f1 --name fold_1 --n_gpus 1

# CPU, primarily for debugging
python model_training/fit.py MODEL_ROOT/f1 --name fold_1 --n_gpus 0
```

Changing GPU count changes batch size, learning rate, optimizer-update count,
gradient noise, and potentially model calibration. Use `--n_gpus 2` when
comparing against the recent ablation runs.

## Checkpoints and stopping

Training runs for at most 20 epochs and monitors `val_loss`.

Two independent checkpoint callbacks are used:

1. The stable best-model checkpoint is checked after every epoch and overwritten
   only when validation loss improves.
2. Periodic checkpoints are retained every epoch, regardless of
   validation loss.

Fresh training produces:

```text
fold_1_best.hdf5
fold_1_epoch_001.hdf5
fold_1_epoch_002.hdf5
...
fold_1_epoch_020.hdf5
```

The early-stopping patience is intentionally longer than the epoch limit, so
fresh runs are expected to complete all 20 epochs unless interrupted manually.
The stable `*_best.hdf5` file remains the lowest-validation-loss checkpoint
encountered during those 20 epochs.

Other outputs include:

```text
fold_1.log
fold_1_history.json
fold_1_architecture.json
```

The CSV log and history JSON contain the epoch losses needed to identify the
selected validation minimum and compare convergence among folds.

Historical 100-epoch runs used checkpoint names such as:

```text
fold_1_epoch_005.hdf5
fold_1_epoch_010.hdf5
...
fold_1_epoch_100.hdf5
```

## Resume training

Resume from an existing checkpoint with:

```bash
python model_training/fit.py \
    MODEL_ROOT/f1 \
    --name fold_1 \
    --resume_checkpoint MODEL_ROOT/f1/fold_1_epoch_050.hdf5
```

Resumed outputs use a separate namespace:

```text
fold_1_resume_best.hdf5
fold_1_resume_epoch_005.hdf5
fold_1_resume_epoch_010.hdf5
...
```

The resumed epoch numbers describe epochs in the new `model.fit()` call; they
are not automatically offset by the epoch encoded in the input filename.
Resume training also creates a fresh optimizer rather than restoring optimizer
state because the model is loaded with `compile=False`.

## Scheduler jobs

A distributed job must request two GPUs for the default protocol. For SLURM,
the exact resource syntax depends on the cluster, but the allocation must expose
two devices to TensorFlow before running:

```bash
python model_training/fit.py MODEL_ROOT/f${SLURM_ARRAY_TASK_ID} \
    --name fold_${SLURM_ARRAY_TASK_ID} \
    --n_gpus 2
```

`bridges2.sh` submits all 27 ablation training jobs as a SLURM array:

```text
clipnet/f1..f9
mean_model/f1..f9
ref_model/f1..f9
```

Each task requests two L48S GPUs and invokes `fit.py --n_gpus 2`. Submit it
after generating every fold's `dataset_params.json`:

```bash
sbatch model_training/bridges2.sh
```

Array indices `0-8` train `clipnet`, `9-17` train `mean_model`, and `18-26`
train `ref_model`. The script expects these directories under `models/` at the
repository root and fails early when a fold configuration is missing.

## Reproducibility checks

For every run, retain:

- The startup line reporting GPU count, effective batch size, learning rate,
  epoch limit, and early-stopping patience.
- TensorFlow/Keras and CUDA versions.
- GPU model and count.
- `dataset_params.json`.
- Training CSV log and history JSON.
- Stable best and periodic checkpoint files.

For QTL comparisons, benchmark both the stable best checkpoint and retained
epoch checkpoints. Use `fold_macro` as the primary within-fold statistic and
inspect pooled-fold correlation and calibration penalty separately; see
[`evaluation_qtl/README.md`](../evaluation_qtl/README.md).

## Source files

- `clipnet_arch.py`: architecture and optimization hyperparameters.
- `clipnet_all_checkpoints.py`: distributed training, callbacks, checkpointing,
  prediction, and model utilities.
- `fit.py`: command-line training entrypoint.
- `calculate_fold_params*.py`: ablation-specific fold configuration generators.

The older `README_archive.md`, `README_reference.md`, and
`README_subsample.md` files are retained as historical experiment notes. This
README documents the current training behavior.
