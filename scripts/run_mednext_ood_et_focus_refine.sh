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
CALIBRATION_GRID="${CALIBRATION_GRID:-0.3,0.4,0.5,0.6,0.7,0.8,0.9}"
FINE_CALIBRATION_GRID="${FINE_CALIBRATION_GRID:-0.3,0.35,0.4,0.45,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7}"
SOURCE_ADAPTED_CHECKPOINT="${SOURCE_ADAPTED_CHECKPOINT:-runs/_ood_adapt_utsw_to_brats_v4_bias_region_e4/best.pt}"
CACHE_DIR="${ET_FOCUS_CACHE_DIR:-runs/cache/_ood_adapt_utsw_to_brats_v4_bias_region_e1}"
CAUSAL_CACHE_DIR="${CAUSAL_ET_FOCUS_CACHE_DIR:-runs/cache/_ood_causal_adapt_brats_v5_et_focus_e1}"

ET_E1_DIR="${ET_E1_DIR:-runs/_ood_adapt_utsw_to_brats_v5_et_focus_e1}"
ET_E2_DIR="${ET_E2_DIR:-runs/_ood_adapt_utsw_to_brats_v5_et_focus_e2}"
ET_E3_DIR="${ET_E3_DIR:-runs/_ood_adapt_utsw_to_brats_v5_et_focus_e3}"
CAUSAL_E1_DIR="${CAUSAL_E1_DIR:-runs/_ood_causal_adapt_brats_v5_et_focus_e1}"
CAUSAL_E2_DIR="${CAUSAL_E2_DIR:-runs/_ood_causal_adapt_brats_v5_et_focus_e2}"
CAUSAL_FINE_EVAL_JSON="${CAUSAL_FINE_EVAL_JSON:-$CAUSAL_E2_DIR/brats_val_fine_thresholds_metrics.json}"
PRECISION_E1_DIR="${PRECISION_E1_DIR:-runs/_ood_causal_adapt_brats_v5_et_precision_e1}"
PRECISION_E2_DIR="${PRECISION_E2_DIR:-runs/_ood_causal_adapt_brats_v5_et_precision_e2}"
STRUCTURAL_EVAL_JSON="${STRUCTURAL_EVAL_JSON:-$PRECISION_E2_DIR/brats_val_structural_min32_metrics.json}"

run_et_stage() {
  local out_dir="$1"
  local init_checkpoint="$2"
  local lr="$3"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/best.pt" ]]; then
    echo "Skipping ET-focused stage; found $out_dir/best.pt"
    return
  fi
  PYTHON_BIN="$PYTHON_BIN" \
  OUT_DIR="$out_dir" \
  INIT_CHECKPOINT="$init_checkpoint" \
  BASELINE_EPOCHS=1 \
  LIMIT_TRAIN_VOLUMES="${LIMIT_TRAIN_VOLUMES:-4}" \
  LIMIT_VAL_VOLUMES="${LIMIT_VAL_VOLUMES:-4}" \
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-4}" \
  MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-4}" \
  CACHE_DIR="$CACHE_DIR" \
  LR="$lr" \
  SEG_LOSS_MODE=balanced_focal \
  CHANNEL_LOSS_WEIGHTS="${CHANNEL_LOSS_WEIGHTS:-1.0,2.0,5.0}" \
  LAMBDA_REGION_LOSS="${LAMBDA_REGION_LOSS:-0.75}" \
  REGION_LOSS_WEIGHTS="${REGION_LOSS_WEIGHTS:-1.0,3.0,7.0}" \
  LAMBDA_VOLUME_PRIOR_LOSS="${LAMBDA_VOLUME_PRIOR_LOSS:-0.05}" \
  CHECKPOINT_CALIBRATION_THRESHOLDS="$CALIBRATION_GRID" \
  bash scripts/train_mednext_brats_paper64_baseline.sh
}

run_causal_stage() {
  local out_dir="$1"
  local init_checkpoint="$2"
  local base_lr="$3"
  local backbone_lr="$4"
  local causal_lr="$5"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/best.pt" ]]; then
    echo "Skipping ET-focused causal stage; found $out_dir/best.pt"
    return
  fi
  PYTHON_BIN="$PYTHON_BIN" \
  BASELINE_CHECKPOINT="$ET_E3_DIR/best.pt" \
  INIT_CHECKPOINT="$init_checkpoint" \
  OUT_DIR="$out_dir" \
  CAUSAL_EPOCHS=1 \
  LIMIT_TRAIN_VOLUMES="${LIMIT_TRAIN_VOLUMES:-4}" \
  LIMIT_VAL_VOLUMES="${LIMIT_VAL_VOLUMES:-4}" \
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-4}" \
  MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-4}" \
  CACHE_DIR="$CAUSAL_CACHE_DIR" \
  CONTEXT_BANK_SIZE="${CONTEXT_BANK_SIZE:-8}" \
  MAX_CONTEXT_BANK_BATCHES="${MAX_CONTEXT_BANK_BATCHES:-4}" \
  ADJUSTMENT_CONTEXTS="${ADJUSTMENT_CONTEXTS:-2}" \
  ADJUSTMENT_CONTEXT_SELECTION="${ADJUSTMENT_CONTEXT_SELECTION:-diverse-nearest}" \
  FREEZE_BACKBONE_EPOCHS=0 \
  LR="$base_lr" \
  BACKBONE_LR="$backbone_lr" \
  CAUSAL_LR="$causal_lr" \
  SEG_LOSS_MODE=balanced_focal \
  CHANNEL_LOSS_WEIGHTS="${CHANNEL_LOSS_WEIGHTS:-1.0,2.0,5.0}" \
  LAMBDA_REGION_LOSS="${LAMBDA_REGION_LOSS:-0.75}" \
  REGION_LOSS_WEIGHTS="${REGION_LOSS_WEIGHTS:-1.0,3.0,7.0}" \
  LAMBDA_VOLUME_PRIOR_LOSS="${LAMBDA_VOLUME_PRIOR_LOSS:-0.05}" \
  CHECKPOINT_CALIBRATION_THRESHOLDS="$CALIBRATION_GRID" \
  bash scripts/train_mednext_brats_paper64_causal.sh
}

run_precision_causal_stage() {
  local out_dir="$1"
  local init_checkpoint="$2"
  local base_lr="$3"
  local backbone_lr="$4"
  local causal_lr="$5"
  local focal_alpha="$6"
  local focal_beta="$7"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/best.pt" ]]; then
    echo "Skipping ET-precision causal stage; found $out_dir/best.pt"
    return
  fi
  PYTHON_BIN="$PYTHON_BIN" \
  BASELINE_CHECKPOINT="$ET_E3_DIR/best.pt" \
  INIT_CHECKPOINT="$init_checkpoint" \
  OUT_DIR="$out_dir" \
  CAUSAL_EPOCHS=1 \
  LIMIT_TRAIN_VOLUMES="${LIMIT_TRAIN_VOLUMES:-4}" \
  LIMIT_VAL_VOLUMES="${LIMIT_VAL_VOLUMES:-4}" \
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-4}" \
  MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-4}" \
  CACHE_DIR="$CAUSAL_CACHE_DIR" \
  CONTEXT_BANK_SIZE="${CONTEXT_BANK_SIZE:-8}" \
  MAX_CONTEXT_BANK_BATCHES="${MAX_CONTEXT_BANK_BATCHES:-4}" \
  ADJUSTMENT_CONTEXTS="${ADJUSTMENT_CONTEXTS:-2}" \
  ADJUSTMENT_CONTEXT_SELECTION="${ADJUSTMENT_CONTEXT_SELECTION:-diverse-nearest}" \
  FREEZE_BACKBONE_EPOCHS=0 \
  LR="$base_lr" \
  BACKBONE_LR="$backbone_lr" \
  CAUSAL_LR="$causal_lr" \
  SEG_LOSS_MODE=balanced_focal \
  CHANNEL_LOSS_WEIGHTS="${CHANNEL_LOSS_WEIGHTS:-1.0,2.0,5.0}" \
  LAMBDA_REGION_LOSS="${LAMBDA_REGION_LOSS:-0.75}" \
  REGION_LOSS_WEIGHTS="${REGION_LOSS_WEIGHTS:-1.0,3.0,7.0}" \
  LAMBDA_VOLUME_PRIOR_LOSS="${LAMBDA_VOLUME_PRIOR_LOSS:-0.05}" \
  FOCAL_TVERSKY_ALPHA="$focal_alpha" \
  FOCAL_TVERSKY_BETA="$focal_beta" \
  CHECKPOINT_CALIBRATION_THRESHOLDS="$FINE_CALIBRATION_GRID" \
  CHECKPOINT_CALIBRATION_OBJECTIVE="${CHECKPOINT_CALIBRATION_OBJECTIVE:-tc_et_min}" \
  bash scripts/train_mednext_brats_paper64_causal.sh
}

if [[ ! -f "$SOURCE_ADAPTED_CHECKPOINT" ]]; then
  echo "Missing source adapted checkpoint: $SOURCE_ADAPTED_CHECKPOINT" >&2
  exit 1
fi

run_et_stage "$ET_E1_DIR" "$SOURCE_ADAPTED_CHECKPOINT" "${ET_E1_LR:-0.0002}"
run_et_stage "$ET_E2_DIR" "$ET_E1_DIR/best.pt" "${ET_E2_LR:-0.0001}"
run_et_stage "$ET_E3_DIR" "$ET_E2_DIR/best.pt" "${ET_E3_LR:-0.00005}"

run_causal_stage "$CAUSAL_E1_DIR" "" "${CAUSAL_E1_BASE_LR:-0.00005}" "${CAUSAL_E1_BACKBONE_LR:-0.00001}" "${CAUSAL_E1_HEAD_LR:-0.0002}"
run_causal_stage "$CAUSAL_E2_DIR" "$CAUSAL_E1_DIR/best.pt" "${CAUSAL_E2_BASE_LR:-0.00002}" "${CAUSAL_E2_BACKBONE_LR:-0.000005}" "${CAUSAL_E2_HEAD_LR:-0.0001}"

if [[ "${RUN_PRECISION_REFINE:-1}" == "1" ]]; then
  run_precision_causal_stage "$PRECISION_E1_DIR" "$CAUSAL_E2_DIR/best.pt" "${PRECISION_E1_BASE_LR:-0.00001}" "${PRECISION_E1_BACKBONE_LR:-0.000002}" "${PRECISION_E1_HEAD_LR:-0.00005}" "${PRECISION_E1_FOCAL_ALPHA:-0.6}" "${PRECISION_E1_FOCAL_BETA:-0.4}"
  run_precision_causal_stage "$PRECISION_E2_DIR" "$PRECISION_E1_DIR/best.pt" "${PRECISION_E2_BASE_LR:-0.000005}" "${PRECISION_E2_BACKBONE_LR:-0.000001}" "${PRECISION_E2_HEAD_LR:-0.000025}" "${PRECISION_E2_FOCAL_ALPHA:-0.55}" "${PRECISION_E2_FOCAL_BETA:-0.45}"
fi

if [[ "$SKIP_EXISTING" == "1" && -f "$CAUSAL_FINE_EVAL_JSON" ]]; then
  echo "Skipping ET-focused causal fine eval; found $CAUSAL_FINE_EVAL_JSON"
else
  PYTHON_BIN="$PYTHON_BIN" \
  CHECKPOINT="$CAUSAL_E2_DIR/best.pt" \
  OUT_JSON="$CAUSAL_FINE_EVAL_JSON" \
  MAX_VOLUMES="${FINE_EVAL_MAX_VOLUMES:-4}" \
  MAX_BATCHES="${FINE_EVAL_MAX_BATCHES:-4}" \
  MAX_CONTEXT_VOLUMES="${FINE_EVAL_MAX_CONTEXT_VOLUMES:-4}" \
  MAX_CONTEXT_BANK_BATCHES="${FINE_EVAL_MAX_CONTEXT_BANK_BATCHES:-4}" \
  CONTEXT_BANK_SIZE="${FINE_EVAL_CONTEXT_BANK_SIZE:-8}" \
  CONTEXT_BANK_SAMPLING="${FINE_EVAL_CONTEXT_BANK_SAMPLING:-farthest}" \
  ADJUSTMENT_CONTEXTS="${FINE_EVAL_ADJUSTMENT_CONTEXTS:-2}" \
  ADJUSTMENT_CONTEXT_SELECTION="${ADJUSTMENT_CONTEXT_SELECTION:-diverse-nearest}" \
  CALIBRATION_THRESHOLDS="$FINE_CALIBRATION_GRID" \
  SPLIT_NAME="${FINE_EVAL_SPLIT_NAME:-ood_causal_v5_et_focus_fine_thresholds_smoke4}" \
  bash scripts/evaluate_mednext_brats_paper64_causal.sh
fi

if [[ "$SKIP_EXISTING" == "1" && -f "$STRUCTURAL_EVAL_JSON" ]]; then
  echo "Skipping ET-precision structural eval; found $STRUCTURAL_EVAL_JSON"
else
  PYTHON_BIN="$PYTHON_BIN" \
  CHECKPOINT="$PRECISION_E2_DIR/best.pt" \
  OUT_JSON="$STRUCTURAL_EVAL_JSON" \
  MAX_VOLUMES="${STRUCTURAL_EVAL_MAX_VOLUMES:-4}" \
  MAX_BATCHES="${STRUCTURAL_EVAL_MAX_BATCHES:-4}" \
  MAX_CONTEXT_VOLUMES="${STRUCTURAL_EVAL_MAX_CONTEXT_VOLUMES:-4}" \
  MAX_CONTEXT_BANK_BATCHES="${STRUCTURAL_EVAL_MAX_CONTEXT_BANK_BATCHES:-4}" \
  CONTEXT_BANK_SIZE="${STRUCTURAL_EVAL_CONTEXT_BANK_SIZE:-8}" \
  CONTEXT_BANK_SAMPLING="${STRUCTURAL_EVAL_CONTEXT_BANK_SAMPLING:-farthest}" \
  ADJUSTMENT_CONTEXTS="${STRUCTURAL_EVAL_ADJUSTMENT_CONTEXTS:-2}" \
  ADJUSTMENT_CONTEXT_SELECTION="${ADJUSTMENT_CONTEXT_SELECTION:-diverse-nearest}" \
  REGION_THRESHOLDS="${STRUCTURAL_REGION_THRESHOLDS:-WT=0.65,TC=0.3,ET=0.55}" \
  STRUCTURAL_PRIOR=1 \
  STRUCTURAL_THRESHOLD="${STRUCTURAL_THRESHOLD:-0.55}" \
  STRUCTURAL_MIN_COMPONENT_SIZE="${STRUCTURAL_MIN_COMPONENT_SIZE:-32}" \
  SPLIT_NAME="${STRUCTURAL_EVAL_SPLIT_NAME:-ood_causal_v5_et_precision_structural_min32_smoke4}" \
  bash scripts/evaluate_mednext_brats_paper64_causal.sh
fi

"$SUMMARY_PYTHON" scripts/summarize_mednext_ood_smoke.py \
  --causal-et-fine-json "$CAUSAL_FINE_EVAL_JSON" \
  --causal-et-precision-json "$PRECISION_E2_DIR/epoch_001.json" \
  --causal-et-structural-json "$STRUCTURAL_EVAL_JSON" \
  --output-md runs/_ood_utsw_to_brats_summary.md
