#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x .venv312_restore/bin/python ]]; then
    PYTHON_BIN=.venv312_restore/bin/python
  elif [[ -x .venv312/bin/python ]]; then
    PYTHON_BIN=.venv312/bin/python
  elif [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=.venv/bin/python
  else
    PYTHON_BIN=python
  fi
fi

SUMMARY_PYTHON="${SUMMARY_PYTHON:-python3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-runs/mednext_utsw_s_k3/best.pt}"
ZERO_SHOT_JSON="${ZERO_SHOT_JSON:-runs/_ood_utsw_best_on_brats_val_smoke2.json}"
ADAPT_STAGE1_DIR="${ADAPT_STAGE1_DIR:-runs/_ood_adapt_utsw_to_brats_v4_bias_region_e1}"
ADAPT_DIR="${ADAPT_DIR:-runs/_ood_adapt_utsw_to_brats_v4_bias_region_e4}"
ADAPT_EVAL_JSON="${ADAPT_EVAL_JSON:-$ADAPT_DIR/brats_val_smoke4_metrics.json}"
CAUSAL_DIR="${CAUSAL_DIR:-runs/_ood_causal_adapt_brats_v4_e1}"
CAUSAL_EPOCH_JSON="${CAUSAL_EPOCH_JSON:-$CAUSAL_DIR/epoch_001.json}"
CCT_EVAL_JSON="${CCT_EVAL_JSON:-$CAUSAL_DIR/brats_val_cct_diverse_nearest_smoke4_metrics.json}"
SUMMARY_MD="${SUMMARY_MD:-runs/_ood_utsw_to_brats_summary.md}"

CALIBRATION_GRID="${CALIBRATION_GRID:-0.3,0.4,0.5,0.6,0.7,0.8,0.9}"
REGION_THRESHOLDS="${REGION_THRESHOLDS:-WT=0.5,TC=0.3,ET=0.3}"
SEG_LOSS_MODE="${SEG_LOSS_MODE:-balanced_focal}"
CHANNEL_LOSS_WEIGHTS="${CHANNEL_LOSS_WEIGHTS:-1.0,2.0,3.0}"
REGION_LOSS_WEIGHTS="${REGION_LOSS_WEIGHTS:-1.0,2.5,4.0}"
LAMBDA_REGION_LOSS="${LAMBDA_REGION_LOSS:-0.5}"
LAMBDA_VOLUME_PRIOR_LOSS="${LAMBDA_VOLUME_PRIOR_LOSS:-0.05}"
ADJUSTMENT_CONTEXT_SELECTION="${ADJUSTMENT_CONTEXT_SELECTION:-uniform}"

should_skip_file() {
  [[ "$SKIP_EXISTING" == "1" && -f "$1" ]]
}

should_skip_dir_checkpoint() {
  [[ "$SKIP_EXISTING" == "1" && -f "$1/best.pt" ]]
}

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "Missing source checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

if should_skip_file "$ZERO_SHOT_JSON"; then
  echo "Skipping zero-shot OOD eval; found $ZERO_SHOT_JSON"
else
  PYTHON_BIN="$PYTHON_BIN" \
  CHECKPOINT="$SOURCE_CHECKPOINT" \
  OUT_JSON="$ZERO_SHOT_JSON" \
  MAX_VOLUMES="${ZERO_SHOT_MAX_VOLUMES:-2}" \
  MAX_BATCHES="${ZERO_SHOT_MAX_BATCHES:-2}" \
  CALIBRATION_THRESHOLDS="$CALIBRATION_GRID" \
  SPLIT_NAME="${ZERO_SHOT_SPLIT_NAME:-ood_utsw_to_brats_smoke}" \
  bash scripts/evaluate_mednext_brats_paper64_baseline.sh
fi

if should_skip_dir_checkpoint "$ADAPT_STAGE1_DIR"; then
  echo "Skipping adaptation stage 1; found $ADAPT_STAGE1_DIR/best.pt"
else
  PYTHON_BIN="$PYTHON_BIN" \
  OUT_DIR="$ADAPT_STAGE1_DIR" \
  INIT_CHECKPOINT="$SOURCE_CHECKPOINT" \
  BASELINE_EPOCHS="${ADAPT_STAGE1_EPOCHS:-1}" \
  LIMIT_TRAIN_VOLUMES="${ADAPT_TRAIN_VOLUMES:-4}" \
  LIMIT_VAL_VOLUMES="${ADAPT_STAGE1_VAL_VOLUMES:-2}" \
  MAX_TRAIN_BATCHES="${ADAPT_MAX_TRAIN_BATCHES:-4}" \
  MAX_VAL_BATCHES="${ADAPT_STAGE1_MAX_VAL_BATCHES:-2}" \
  CACHE_DIR="${ADAPT_CACHE_DIR:-runs/cache/_ood_adapt_utsw_to_brats_v4_bias_region_e1}" \
  LR="${ADAPT_LR:-0.0005}" \
  INIT_OUTPUT_BIAS_FROM_DATA=1 \
  SEG_LOSS_MODE="$SEG_LOSS_MODE" \
  CHANNEL_LOSS_WEIGHTS="$CHANNEL_LOSS_WEIGHTS" \
  LAMBDA_REGION_LOSS="$LAMBDA_REGION_LOSS" \
  REGION_LOSS_WEIGHTS="$REGION_LOSS_WEIGHTS" \
  LAMBDA_VOLUME_PRIOR_LOSS="$LAMBDA_VOLUME_PRIOR_LOSS" \
  CHECKPOINT_CALIBRATION_THRESHOLDS="$CALIBRATION_GRID" \
  bash scripts/train_mednext_brats_paper64_baseline.sh
fi

if should_skip_dir_checkpoint "$ADAPT_DIR"; then
  echo "Skipping adaptation continuation; found $ADAPT_DIR/best.pt"
else
  PYTHON_BIN="$PYTHON_BIN" \
  OUT_DIR="$ADAPT_DIR" \
  INIT_CHECKPOINT="$ADAPT_STAGE1_DIR/best.pt" \
  BASELINE_EPOCHS="${ADAPT_STAGE2_EPOCHS:-3}" \
  LIMIT_TRAIN_VOLUMES="${ADAPT_TRAIN_VOLUMES:-4}" \
  LIMIT_VAL_VOLUMES="${ADAPT_STAGE2_VAL_VOLUMES:-2}" \
  MAX_TRAIN_BATCHES="${ADAPT_MAX_TRAIN_BATCHES:-4}" \
  MAX_VAL_BATCHES="${ADAPT_STAGE2_MAX_VAL_BATCHES:-2}" \
  CACHE_DIR="${ADAPT_CACHE_DIR:-runs/cache/_ood_adapt_utsw_to_brats_v4_bias_region_e1}" \
  LR="${ADAPT_LR:-0.0005}" \
  SEG_LOSS_MODE="$SEG_LOSS_MODE" \
  CHANNEL_LOSS_WEIGHTS="$CHANNEL_LOSS_WEIGHTS" \
  LAMBDA_REGION_LOSS="$LAMBDA_REGION_LOSS" \
  REGION_LOSS_WEIGHTS="$REGION_LOSS_WEIGHTS" \
  LAMBDA_VOLUME_PRIOR_LOSS="$LAMBDA_VOLUME_PRIOR_LOSS" \
  CHECKPOINT_CALIBRATION_THRESHOLDS="$CALIBRATION_GRID" \
  bash scripts/train_mednext_brats_paper64_baseline.sh
fi

if should_skip_file "$ADAPT_EVAL_JSON"; then
  echo "Skipping adapted checkpoint eval; found $ADAPT_EVAL_JSON"
else
  PYTHON_BIN="$PYTHON_BIN" \
  CHECKPOINT="$ADAPT_DIR/best.pt" \
  OUT_JSON="$ADAPT_EVAL_JSON" \
  MAX_VOLUMES="${ADAPT_EVAL_MAX_VOLUMES:-4}" \
  MAX_BATCHES="${ADAPT_EVAL_MAX_BATCHES:-4}" \
  CALIBRATION_THRESHOLDS="$CALIBRATION_GRID" \
  SPLIT_NAME="${ADAPT_EVAL_SPLIT_NAME:-ood_adapt_brats_val_smoke4}" \
  bash scripts/evaluate_mednext_brats_paper64_baseline.sh
fi

if should_skip_dir_checkpoint "$CAUSAL_DIR"; then
  echo "Skipping causal SCM adaptation; found $CAUSAL_DIR/best.pt"
else
  PYTHON_BIN="$PYTHON_BIN" \
  BASELINE_CHECKPOINT="$ADAPT_DIR/best.pt" \
  OUT_DIR="$CAUSAL_DIR" \
  CAUSAL_EPOCHS="${CAUSAL_EPOCHS:-1}" \
  LIMIT_TRAIN_VOLUMES="${CAUSAL_TRAIN_VOLUMES:-4}" \
  LIMIT_VAL_VOLUMES="${CAUSAL_VAL_VOLUMES:-4}" \
  MAX_TRAIN_BATCHES="${CAUSAL_MAX_TRAIN_BATCHES:-4}" \
  MAX_VAL_BATCHES="${CAUSAL_MAX_VAL_BATCHES:-4}" \
  CACHE_DIR="${CAUSAL_CACHE_DIR:-runs/cache/_ood_causal_adapt_brats_v4_e1}" \
  CONTEXT_BANK_SIZE="${CONTEXT_BANK_SIZE:-8}" \
  MAX_CONTEXT_BANK_BATCHES="${MAX_CONTEXT_BANK_BATCHES:-4}" \
  ADJUSTMENT_CONTEXTS="${ADJUSTMENT_CONTEXTS:-2}" \
  ADJUSTMENT_CONTEXT_SELECTION="$ADJUSTMENT_CONTEXT_SELECTION" \
  FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0}" \
  LR="${CAUSAL_BASE_LR:-0.00005}" \
  BACKBONE_LR="${CAUSAL_BACKBONE_LR:-0.00001}" \
  CAUSAL_LR="${CAUSAL_HEAD_LR:-0.0002}" \
  SEG_LOSS_MODE="$SEG_LOSS_MODE" \
  CHANNEL_LOSS_WEIGHTS="$CHANNEL_LOSS_WEIGHTS" \
  LAMBDA_REGION_LOSS="$LAMBDA_REGION_LOSS" \
  REGION_LOSS_WEIGHTS="$REGION_LOSS_WEIGHTS" \
  LAMBDA_VOLUME_PRIOR_LOSS="$LAMBDA_VOLUME_PRIOR_LOSS" \
  CHECKPOINT_CALIBRATION_THRESHOLDS="$CALIBRATION_GRID" \
  bash scripts/train_mednext_brats_paper64_causal.sh
fi

if [[ "${RUN_CCT_EVAL:-0}" == "1" ]]; then
  if should_skip_file "$CCT_EVAL_JSON"; then
    echo "Skipping CCT eval; found $CCT_EVAL_JSON"
  else
    PYTHON_BIN="$PYTHON_BIN" \
    CHECKPOINT="$CAUSAL_DIR/best.pt" \
    OUT_JSON="$CCT_EVAL_JSON" \
    MAX_VOLUMES="${CCT_MAX_VOLUMES:-4}" \
    MAX_BATCHES="${CCT_MAX_BATCHES:-4}" \
    MAX_CONTEXT_VOLUMES="${CCT_MAX_CONTEXT_VOLUMES:-4}" \
    MAX_CONTEXT_BANK_BATCHES="${CCT_MAX_CONTEXT_BANK_BATCHES:-4}" \
    CONTEXT_BANK_SIZE="${CCT_CONTEXT_BANK_SIZE:-8}" \
    CONTEXT_BANK_SAMPLING="${CCT_CONTEXT_BANK_SAMPLING:-farthest}" \
    ADJUSTMENT_CONTEXTS="${CCT_ADJUSTMENT_CONTEXTS:-2}" \
    ADJUSTMENT_CONTEXT_SELECTION="${CCT_ADJUSTMENT_CONTEXT_SELECTION:-$ADJUSTMENT_CONTEXT_SELECTION}" \
    CCT_CONTEXTS="${CCT_CONTEXTS:-4}" \
    CCT_SELECTION="${CCT_SELECTION:-diverse-nearest}" \
    REGION_THRESHOLDS="$REGION_THRESHOLDS" \
    CALIBRATION_THRESHOLDS="$CALIBRATION_GRID" \
    SPLIT_NAME="${CCT_SPLIT_NAME:-ood_causal_cct_brats_val_smoke4}" \
    bash scripts/evaluate_mednext_brats_paper64_causal.sh
  fi
fi

"$SUMMARY_PYTHON" scripts/summarize_mednext_ood_smoke.py \
  --zero-shot-json "$ZERO_SHOT_JSON" \
  --adapted-eval-json "$ADAPT_EVAL_JSON" \
  --causal-epoch-json "$CAUSAL_EPOCH_JSON" \
  --cct-eval-json "$CCT_EVAL_JSON" \
  --output-md "$SUMMARY_MD"
