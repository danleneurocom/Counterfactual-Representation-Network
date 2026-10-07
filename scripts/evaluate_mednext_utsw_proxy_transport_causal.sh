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

CHECKPOINT="${CHECKPOINT:-runs/mednext_utsw_causal_proxy/best.pt}"
OUT_JSON="${OUT_JSON:-runs/mednext_utsw_causal_proxy/utsw_val_causal_metrics.json}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEFAULT_METADATA_PATH="${DEFAULT_METADATA_PATH:-data/brats/UTSW_Glioma_Metadata-2-1.tsv}"
METADATA_PATH="${METADATA_PATH:-$DEFAULT_METADATA_PATH}"

cmd=(
  "$PYTHON_BIN" baselines/mednext/evaluate_causal_utsw.py
  --checkpoint "$CHECKPOINT"
  --split "${SPLIT:-val}"
  --context-split "${CONTEXT_SPLIT:-train}"
  --data-root "${DATA_ROOT:-data/brats/PKG - UTSW-Glioma/UTSW-Glioma}"
  --output-json "$OUT_JSON"
  --context-bank-size "${CONTEXT_BANK_SIZE:-64}"
  --context-bank-sampling "${CONTEXT_BANK_SAMPLING:-farthest}"
  --adjustment-contexts "${ADJUSTMENT_CONTEXTS:-2}"
  --threshold "${THRESHOLD:-0.5}"
  --num-workers "$NUM_WORKERS"
)

if [[ "${PREFER_MANUAL_SEG:-1}" == "1" ]]; then
  cmd+=(--prefer-manual-seg)
fi
if [[ "${USE_ANTS_MODALITIES:-0}" == "1" ]]; then
  cmd+=(--use-ants-modalities)
fi
if [[ -n "$METADATA_PATH" && -f "$METADATA_PATH" ]]; then
  cmd+=(--metadata-path "$METADATA_PATH")
else
  cmd+=(--allow-missing-metadata)
fi
if [[ -n "${MAX_CONTEXT_BANK_BATCHES:-}" ]]; then
  cmd+=(--max-context-bank-batches "$MAX_CONTEXT_BANK_BATCHES")
fi
if [[ -n "${MAX_BATCHES:-}" ]]; then
  cmd+=(--max-batches "$MAX_BATCHES")
fi
if [[ -n "${ADJUSTMENT_CONTEXT_SELECTION:-}" ]]; then
  cmd+=(--adjustment-context-selection "$ADJUSTMENT_CONTEXT_SELECTION")
fi
if [[ -n "${CALIBRATION_THRESHOLDS:-}" ]]; then
  cmd+=(--calibration-thresholds "$CALIBRATION_THRESHOLDS")
fi
if [[ -n "${CALIBRATION_OBJECTIVE:-}" ]]; then
  cmd+=(--calibration-objective "$CALIBRATION_OBJECTIVE")
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

"${cmd[@]}"
