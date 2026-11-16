#!/bin/bash
# Score the validation-loss-selected checkpoints (valloss_selection.py) and
# bootstrap clipnet - ref_model for the 20- and 100-epoch runs.
#   bash run_valloss_selection.sh        # from evaluation_qtl/
set -euo pipefail
PY="${PY:-${HOME}/miniconda3/envs/clipnet/bin/python}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-/fs/cbsubscb17/storage/projects/CLIPNET_ablation/revision_predictions}"
DATA_ROOT="${DATA_ROOT:-../data}"
OUT="${OUT:-nonretraining_results}"
UNIVERSE=epoch_checkpoint_benchmark/clipnet/scores_published_l2/epoch_001_l2_scores.csv.gz
cd "$(dirname "$0")"
"${PY}" valloss_selection.py
for qtl in tiqtl diqtl; do
    for run in clipnet ref_model five_epoch_clipnet ref100_ref_model; do
        "${PY}" alternative_qtl_scores.py all \
            "${PREDICTIONS_ROOT}/${qtl}/epoch_checkpoint_benchmark/${run}_valsel/folds" \
            --qtl "${qtl}" --qtl_data_dir "${DATA_ROOT}/${qtl}" \
            --universe "${PREDICTIONS_ROOT}/${qtl}/${UNIVERSE}" --workers 16 --skip_existing
    done
    for pair in "valsel20:clipnet:ref_model" "valsel100:five_epoch_clipnet:ref100_ref_model"; do
        IFS=: read -r label clip ref <<<"${pair}"
        for metric in published_l2 published_l2_partial shape_l2 expression_only; do
            "${PY}" bootstrap_qtl_ci.py \
                --entry "clipnet=${PREDICTIONS_ROOT}/${qtl}/epoch_checkpoint_benchmark/${clip}_valsel/folds/alt_scores/${metric}@valsel" \
                --entry "ref_model=${PREDICTIONS_ROOT}/${qtl}/epoch_checkpoint_benchmark/${ref}_valsel/folds/alt_scores/${metric}@valsel" \
                --output_dir "${OUT}/${qtl}" --prefix "${label}_${metric}" \
                >"${OUT}/${qtl}/${label}_${metric}.log" 2>&1
        done
    done
done
echo "VALSEL DONE"
