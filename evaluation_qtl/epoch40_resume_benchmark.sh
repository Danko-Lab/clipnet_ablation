#!/bin/bash

# QTL benchmarks for the epoch-40 -> epoch-45 resume experiment.
# Run from evaluation_qtl/ on a GPU node after model_training/epoch40_resume.sh
# finishes. Each replicate is scored in --mode folds, so its fold_macro r is
# directly comparable to the epoch traces.

set -euo pipefail

RESUME_ROOT="${RESUME_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/models/epoch40_resume}"
MAIN_ROOT="${MAIN_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-../predictions}"
DATA_ROOT="${DATA_ROOT:-../data}"
QTL_ARCHIVE="${QTL_ARCHIVE:-${DATA_ROOT}/qtl_analysis.tar.gz}"
EPOCHS="${EPOCHS:-45}"  # use 41,42,43,44,45 for full replicate trajectories
N_REPS="${N_REPS:-20}"

# 1. Score epoch 45 of every replicate.
for qtl in tiqtl diqtl; do
    for arm in clipnet ref_model; do
        for rep in $(seq -f "%02g" 1 "${N_REPS}"); do
            python benchmark_epoch_checkpoints.py all "${RESUME_ROOT}/${arm}/rep${rep}" \
                --qtl "${qtl}" \
                --mode folds \
                --epochs "${EPOCHS}" \
                --run_name "e40resume_${arm}_rep${rep}" \
                --predictions_root "${PREDICTIONS_ROOT}" \
                --data_root "${DATA_ROOT}" \
                --qtl_data_dir "${DATA_ROOT}/${qtl}" \
                --experimental_l2_archive "${QTL_ARCHIVE}" \
                --skip_existing \
                --gpu
        done
    done
done

# 2. Control: score the main (early-stopped, best-val-loss) models with the same
#    pipeline, so main personal vs main reference uses the identical metric.
for qtl in tiqtl diqtl; do
    for models in ensemble_models reference_models; do
        python benchmark_published_clipnet.py all "${MAIN_ROOT}/${models}" \
            --qtl "${qtl}" \
            --mode composite \
            --run_name "main_${models}" \
            --predictions_root "${PREDICTIONS_ROOT}" \
            --data_root "${DATA_ROOT}" \
            --qtl_data_dir "${DATA_ROOT}/${qtl}" \
            --experimental_l2_archive "${QTL_ARCHIVE}" \
            --skip_existing \
            --gpu
    done
done

# 3. Violin plot and tests.
for qtl in tiqtl diqtl; do
    python summarize_epoch40_resume.py \
        --predictions_root "${PREDICTIONS_ROOT}" \
        --qtl "${qtl}" \
        --n_reps "${N_REPS}" \
        --baseline_stats "../predictions/_old/${qtl}_summaries/qtl_benchmark_stats.csv" \
        --output_dir "${qtl}_epoch40_resume"
done
