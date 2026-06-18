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

CHECKPOINT="${CHECKPOINT:-runs/mednext_brats_h5_s_k3_paper64/best.pt}"
OUT_JSON="${OUT_JSON:-runs/mednext_brats_h5_s_k3_paper64/brats_val_metrics.json}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cmd=(
  "$PYTHON_BIN" baselines/mednext/evaluate_brats_h5.py
  --checkpoint "$CHECKPOINT"
  --brats-csv "${BRATS_CSV:-data/brats/brats_val.csv}"
  --data-root "${DATA_ROOT:-data/brats/archive/BraTS2020_training_data/content/data}"
  --split-name "${SPLIT_NAME:-brats_val}"
  --output-json "$OUT_JSON"
  --threshold "${THRESHOLD:-0.5}"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "${MAX_VOLUMES:-}" ]]; then
  cmd+=(--max-volumes "$MAX_VOLUMES")
fi
if [[ -n "${MAX_BATCHES:-}" ]]; then
  cmd+=(--max-batches "$MAX_BATCHES")
fi
if [[ -n "${CALIBRATION_THRESHOLDS:-}" ]]; then
  cmd+=(--calibration-thresholds "$CALIBRATION_THRESHOLDS")
fi
if [[ -n "${CALIBRATION_OBJECTIVE:-}" ]]; then
  cmd+=(--calibration-objective "$CALIBRATION_OBJECTIVE")
fi
if [[ -n "${REGION_THRESHOLDS:-}" ]]; then
  cmd+=(--region-thresholds "$REGION_THRESHOLDS")
fi
if [[ -n "${MIRROR_TTA_AXES:-}" ]]; then
  cmd+=(--mirror-tta-axes "$MIRROR_TTA_AXES")
fi

"${cmd[@]}"
