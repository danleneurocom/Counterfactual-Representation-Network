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

CHECKPOINT="${CHECKPOINT:-runs/mednext_brats_h5_causal_s_k3_paper64/best.pt}"
OUT_JSON="${OUT_JSON:-runs/mednext_brats_h5_causal_s_k3_paper64/brats_val_causal_metrics.json}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cmd=(
  "$PYTHON_BIN" baselines/mednext/evaluate_causal_brats_h5.py
  --checkpoint "$CHECKPOINT"
  --brats-csv "${BRATS_CSV:-data/brats/brats_val.csv}"
  --context-csv "${CONTEXT_CSV:-data/brats/brats_train.csv}"
  --data-root "${DATA_ROOT:-data/brats/archive/BraTS2020_training_data/content/data}"
  --split-name "${SPLIT_NAME:-brats_val}"
  --context-bank-size "${CONTEXT_BANK_SIZE:-64}"
  --context-bank-sampling "${CONTEXT_BANK_SAMPLING:-farthest}"
  --adjustment-contexts "${ADJUSTMENT_CONTEXTS:-2}"
  --threshold "${THRESHOLD:-0.5}"
  --output-json "$OUT_JSON"
  --num-workers "$NUM_WORKERS"
)

if [[ -n "${MAX_CONTEXT_BANK_BATCHES:-}" ]]; then
  cmd+=(--max-context-bank-batches "$MAX_CONTEXT_BANK_BATCHES")
fi
if [[ -n "${MAX_CONTEXT_VOLUMES:-}" ]]; then
  cmd+=(--max-context-volumes "$MAX_CONTEXT_VOLUMES")
fi
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
if [[ "${STRUCTURAL_PRIOR:-0}" == "1" ]]; then
  cmd+=(--structural-prior)
fi
if [[ -n "${STRUCTURAL_THRESHOLD:-}" ]]; then
  cmd+=(--structural-threshold "$STRUCTURAL_THRESHOLD")
fi
if [[ -n "${STRUCTURAL_MIN_COMPONENT_SIZE:-}" ]]; then
  cmd+=(--structural-min-component-size "$STRUCTURAL_MIN_COMPONENT_SIZE")
fi
if [[ "${STRUCTURAL_FILL_HOLES:-0}" == "1" ]]; then
  cmd+=(--structural-fill-holes)
fi
if [[ "${STRUCTURAL_KEEP_LARGEST:-0}" == "1" ]]; then
  cmd+=(--structural-keep-largest)
fi
if [[ -n "${ADJUSTMENT_CONTEXT_SELECTION:-}" ]]; then
  cmd+=(--adjustment-context-selection "$ADJUSTMENT_CONTEXT_SELECTION")
fi
if [[ -n "${MIRROR_TTA_AXES:-}" ]]; then
  cmd+=(--mirror-tta-axes "$MIRROR_TTA_AXES")
fi
if [[ -n "${CCT_CONTEXTS:-}" ]]; then
  cmd+=(--cct-contexts "$CCT_CONTEXTS")
fi
if [[ -n "${CCT_SELECTION:-}" ]]; then
  cmd+=(--cct-selection "$CCT_SELECTION")
fi
if [[ -n "${CCT_INSTABILITY_SCALE:-}" ]]; then
  cmd+=(--cct-instability-scale "$CCT_INSTABILITY_SCALE")
fi
if [[ -n "${CCT_INSTABILITY_THRESHOLD:-}" ]]; then
  cmd+=(--cct-instability-threshold "$CCT_INSTABILITY_THRESHOLD")
fi
if [[ "${INCLUDE_PER_CASE:-0}" == "1" ]]; then
  cmd+=(--include-per-case)
fi

"${cmd[@]}"
