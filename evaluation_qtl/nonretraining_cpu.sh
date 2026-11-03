#!/bin/bash

# CPU half of the non-retraining clipnet vs ref_model analyses. For every
# comparison set whose fold-mode predictions exist:
#   1. alternative_qtl_scores.py: fold-0 ensemble, shape-only L2,
#      coverage-partial L2, expression-only baselines, checkpoint-window
#      ensembles, zero-L2 audit;
#   2. bootstrap_qtl_ci.py: paired SNP-bootstrap CIs for clipnet - ref_model
#      on each metric.
# Sets whose GPU predictions (nonretraining_gpu_queue.sh) are not finished are
# skipped, so rerun this after the GPU jobs end; finished summaries are reused.
#
#   screen -S qtl_cpu bash nonretraining_cpu.sh        # from evaluation_qtl/

set -euo pipefail

PY="${PY:-${HOME}/miniconda3/envs/clipnet/bin/python}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/revision_predictions}"
DATA_ROOT="${DATA_ROOT:-../data}"
CLAIMS="${PREDICTIONS_ROOT}/.nonretraining_claims"
OUT="${OUT:-nonretraining_results}"
WORKERS="${WORKERS:-16}"
N_BOOT="${N_BOOT:-10000}"
METRICS=(published_l2 published_l2_partial shape_l2 expression_only)
# Every scored SNP, for the fold-0 ensemble (identical across runs).
UNIVERSE=epoch_checkpoint_benchmark/clipnet/scores_published_l2/epoch_001_l2_scores.csv.gz

cd "$(dirname "$0")"

# label|clipnet bench dir|ref bench dir|checkpoint (empty = all)|windows|GPU job
SETS=(
    "epochs20|epoch_checkpoint_benchmark/clipnet/folds|epoch_checkpoint_benchmark/ref_model/folds||1-5,6-10,11-15,16-20,1-20|"
    "best_model|best_model_benchmark/clipnet/folds|best_model_benchmark/ref_model/folds|best_model||"
    "main|published_clipnet_benchmark/main_ensemble_models/folds|published_clipnet_benchmark/main_reference_models/folds|published_model||main_reference_models"
    "epochs100|epoch_checkpoint_benchmark/five_epoch_clipnet/folds|epoch_checkpoint_benchmark/five_epoch_ref_model/folds||5-25,30-50,55-75,80-100,5-100,5-15|five_epoch_ref_model"
)

ready() {  # both GPU jobs for this set and QTL have finished (or none needed)
    local job=$1 qtl=$2
    [[ -z "${job}" ]] && return 0
    [[ -f "${CLAIMS}/${job}_${qtl}/done" ]] || return 1
    [[ "${job}" != five_epoch_ref_model ]] || [[ -f "${CLAIMS}/five_epoch_clipnet_${qtl}/done" ]] || return 1
    [[ "${job}" != main_reference_models ]] || [[ -f "${CLAIMS}/main_ensemble_models_${qtl}/done" ]]
}

for qtl in tiqtl diqtl; do
    for set in "${SETS[@]}"; do
        IFS='|' read -r label clip_dir ref_dir checkpoint windows job <<<"${set}"
        mkdir -p "${OUT}/${qtl}"
        if ! ready "${job}" "${qtl}"; then
            echo "[${qtl} ${label}] GPU predictions not finished; skipping."
            continue
        fi
        for dir in "${clip_dir}" "${ref_dir}"; do
            echo "[${qtl} ${label}] alternative scores for ${dir}"
            "${PY}" alternative_qtl_scores.py all "${PREDICTIONS_ROOT}/${qtl}/${dir}" \
                --qtl "${qtl}" --qtl_data_dir "${DATA_ROOT}/${qtl}" \
                --universe "${PREDICTIONS_ROOT}/${qtl}/${UNIVERSE}" \
                ${windows:+--windows "${windows}"} --workers "${WORKERS}" --skip_existing
        done
        for metric in "${METRICS[@]}"; do
            suffix="${checkpoint:+@${checkpoint}}"
            "${PY}" bootstrap_qtl_ci.py \
                --entry "clipnet=${PREDICTIONS_ROOT}/${qtl}/${clip_dir}/alt_scores/${metric}${suffix}" \
                --entry "ref_model=${PREDICTIONS_ROOT}/${qtl}/${ref_dir}/alt_scores/${metric}${suffix}" \
                --n_boot "${N_BOOT}" \
                --output_dir "${OUT}/${qtl}" \
                --prefix "${label}_${metric}" >"${OUT}/${qtl}/${label}_${metric}.log" 2>&1 \
                || echo "bootstrap failed: ${qtl} ${label} ${metric}; see ${OUT}/${qtl}/${label}_${metric}.log"
        done
        for dir in "${clip_dir}" "${ref_dir}"; do
            cp "${PREDICTIONS_ROOT}/${qtl}/${dir}/alt_scores/zero_l2.csv" \
                "${OUT}/${qtl}/${label}_$(basename "$(dirname "${dir}")")_zero_l2.csv"
        done
    done
done

"${PY}" summarize_nonretraining.py --results_dir "${OUT}"
