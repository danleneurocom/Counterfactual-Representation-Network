#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x .venv312/bin/python ]]; then
    PYTHON_BIN=.venv312/bin/python
  elif [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=.venv/bin/python
  else
    PYTHON_BIN=python
  fi
fi

export PYTHONPATH=".:src${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${OUT_DIR:-runs/mednext_brats_h5_s_k3_paper64}"
EPOCHS="${BASELINE_EPOCHS:-${EPOCHS:-50}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CACHE_DIR="${CACHE_DIR:-runs/cache/brats_h5_paper64_v64}"

cmd=(
  "$PYTHON_BIN" baselines/mednext/train_brats_h5.py
  --train-csv "${TRAIN_CSV:-data/brats/brats_train.csv}"
  --val-csv "${VAL_CSV:-data/brats/brats_val.csv}"
  --data-root "${DATA_ROOT:-data/brats/archive/BraTS2020_training_data/content/data}"
  --output-dir "$OUT_DIR"
  --model-id "${MODEL_ID:-S}"
  --kernel-size "${KERNEL_SIZE:-3}"
  --volume-size "${VOLUME_SIZE:-64}"
  --epochs "$EPOCHS"
  --batch-size "${BATCH_SIZE:-1}"
  --lr "${LR:-0.001}"
  --weight-decay "${WEIGHT_DECAY:-0.0001}"
  --threshold "${THRESHOLD:-0.5}"
  --crop-margin "${CROP_MARGIN:-8}"
  --disk-cache-dir "$CACHE_DIR"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  cmd+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [[ -n "${LIMIT_TRAIN_VOLUMES:-}" ]]; then
  cmd+=(--limit-train-volumes "$LIMIT_TRAIN_VOLUMES")
fi
if [[ -n "${LIMIT_VAL_VOLUMES:-}" ]]; then
  cmd+=(--limit-val-volumes "$LIMIT_VAL_VOLUMES")
fi
if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  cmd+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "${MAX_VAL_BATCHES:-}" ]]; then
  cmd+=(--max-val-batches "$MAX_VAL_BATCHES")
fi
if [[ -n "${SEG_LOSS_MODE:-}" ]]; then
  cmd+=(--seg-loss-mode "$SEG_LOSS_MODE")
fi
if [[ -n "${CHANNEL_LOSS_WEIGHTS:-}" ]]; then
  cmd+=(--channel-loss-weights "$CHANNEL_LOSS_WEIGHTS")
fi
if [[ -n "${REGION_LOSS_WEIGHTS:-}" ]]; then
  cmd+=(--region-loss-weights "$REGION_LOSS_WEIGHTS")
fi
if [[ -n "${LAMBDA_REGION_LOSS:-}" ]]; then
  cmd+=(--lambda-region-loss "$LAMBDA_REGION_LOSS")
fi
if [[ -n "${LAMBDA_VOLUME_PRIOR_LOSS:-}" ]]; then
  cmd+=(--lambda-volume-prior-loss "$LAMBDA_VOLUME_PRIOR_LOSS")
fi
if [[ -n "${BALANCED_BCE_MAX_POS_WEIGHT:-}" ]]; then
  cmd+=(--balanced-bce-max-pos-weight "$BALANCED_BCE_MAX_POS_WEIGHT")
fi
if [[ -n "${FOCAL_TVERSKY_ALPHA:-}" ]]; then
  cmd+=(--focal-tversky-alpha "$FOCAL_TVERSKY_ALPHA")
fi
if [[ -n "${FOCAL_TVERSKY_BETA:-}" ]]; then
  cmd+=(--focal-tversky-beta "$FOCAL_TVERSKY_BETA")
fi
if [[ -n "${FOCAL_TVERSKY_GAMMA:-}" ]]; then
  cmd+=(--focal-tversky-gamma "$FOCAL_TVERSKY_GAMMA")
fi
if [[ "${INIT_OUTPUT_BIAS_FROM_DATA:-0}" == "1" ]]; then
  cmd+=(--init-output-bias-from-data)
fi
if [[ -n "${CHECKPOINT_CALIBRATION_THRESHOLDS:-}" ]]; then
  cmd+=(--checkpoint-calibration-thresholds "$CHECKPOINT_CALIBRATION_THRESHOLDS")
fi
if [[ -n "${CHECKPOINT_CALIBRATION_OBJECTIVE:-}" ]]; then
  cmd+=(--checkpoint-calibration-objective "$CHECKPOINT_CALIBRATION_OBJECTIVE")
fi

"${cmd[@]}"
