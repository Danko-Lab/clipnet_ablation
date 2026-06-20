#!/bin/bash

# Commands for validating retrained CLIPNET ablation QTL benchmarks.
# Edit these roots for the remote server, then run commands section by section.

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-../models}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-../predictions}"
DATA_ROOT="${DATA_ROOT:-../data}"
PUBLISHED_MODELS="${PUBLISHED_MODELS:-../clipnet/clipnet_models}"
QTL_ARCHIVE="${QTL_ARCHIVE:-${DATA_ROOT}/qtl_analysis.tar.gz}"

RUNS=(clipnet mean_model ref_model)
QTLS=(tiqtl diqtl)

# 1. Regenerate fold-mode best-model summaries in clean namespaces.
#    Use this when you want to avoid overwriting older benchmark outputs.
for qtl in "${QTLS[@]}"; do
    for run in "${RUNS[@]}"; do
        python evaluation_qtl/benchmark_best_model.py all "${MODELS_ROOT}/${run}" \
            --qtl "${qtl}" \
            --mode folds \
            --run_name "${run}_2gpu_retrain" \
            --predictions_root "${PREDICTIONS_ROOT}" \
            --data_root "${DATA_ROOT}" \
            --qtl_data_dir "${DATA_ROOT}/${qtl}" \
            --experimental_l2_archive "${QTL_ARCHIVE}" \
            --gpu
    done
done

# 2. Regenerate ensemble-mode best-model summaries in the same clean namespaces.
for qtl in "${QTLS[@]}"; do
    for run in "${RUNS[@]}"; do
        python evaluation_qtl/benchmark_best_model.py all "${MODELS_ROOT}/${run}" \
            --qtl "${qtl}" \
            --mode ensemble \
            --run_name "${run}_2gpu_retrain" \
            --predictions_root "${PREDICTIONS_ROOT}" \
            --data_root "${DATA_ROOT}" \
            --qtl_data_dir "${DATA_ROOT}/${qtl}" \
            --experimental_l2_archive "${QTL_ARCHIVE}" \
            --gpu
    done
done

# 3. Regenerate the published CLIPNET control with composite fold metrics.
for qtl in "${QTLS[@]}"; do
    python evaluation_qtl/benchmark_published_clipnet.py score "${PUBLISHED_MODELS}" \
        --qtl "${qtl}" \
        --mode composite \
        --predictions_root "${PREDICTIONS_ROOT}" \
        --data_root "${DATA_ROOT}" \
        --qtl_data_dir "${DATA_ROOT}/${qtl}" \
        --experimental_l2_archive "${QTL_ARCHIVE}"
done

# 4. Audit model roots, histories, fold pathologies, and summaries.
for qtl in "${QTLS[@]}"; do
    python evaluation_qtl/diagnose_retrained_qtl.py \
        --models_root "${MODELS_ROOT}" \
        --predictions_root "${PREDICTIONS_ROOT}" \
        --qtl "${qtl}" \
        --runs clipnet:clipnet_2gpu_retrain,mean_model:mean_model_2gpu_retrain,ref_model:ref_model_2gpu_retrain \
        --output_dir "evaluation_qtl/${qtl}_retrain_diagnostics"
done

# 5. Build readable aggregate reports and plots.
python evaluation_qtl/summarize_qtl_benchmarks.py \
    published="${PREDICTIONS_ROOT}/tiqtl/published_clipnet_benchmark/zenodo_10408623/folds/published_clipnet_qtl_benchmark_summary_published_l2.csv" \
    clipnet="${PREDICTIONS_ROOT}/tiqtl/best_model_benchmark/clipnet_2gpu_retrain/folds/best_model_qtl_benchmark_summary_published_l2.csv" \
    mean_model="${PREDICTIONS_ROOT}/tiqtl/best_model_benchmark/mean_model_2gpu_retrain/folds/best_model_qtl_benchmark_summary_published_l2.csv" \
    ref_model="${PREDICTIONS_ROOT}/tiqtl/best_model_benchmark/ref_model_2gpu_retrain/folds/best_model_qtl_benchmark_summary_published_l2.csv" \
    --output_dir evaluation_qtl/tiqtl_retrain_summaries

python evaluation_qtl/summarize_qtl_benchmarks.py \
    published="${PREDICTIONS_ROOT}/diqtl/published_clipnet_benchmark/zenodo_10408623/folds/published_clipnet_qtl_benchmark_summary_published_l2.csv" \
    clipnet="${PREDICTIONS_ROOT}/diqtl/best_model_benchmark/clipnet_2gpu_retrain/folds/best_model_qtl_benchmark_summary_published_l2.csv" \
    mean_model="${PREDICTIONS_ROOT}/diqtl/best_model_benchmark/mean_model_2gpu_retrain/folds/best_model_qtl_benchmark_summary_published_l2.csv" \
    ref_model="${PREDICTIONS_ROOT}/diqtl/best_model_benchmark/ref_model_2gpu_retrain/folds/best_model_qtl_benchmark_summary_published_l2.csv" \
    --output_dir evaluation_qtl/diqtl_retrain_summaries
