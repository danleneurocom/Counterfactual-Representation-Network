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

BASELINE_DIR="${BASELINE_DIR:-runs/mednext_utsw_s_k3}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$BASELINE_DIR/best.pt}"
OUT_DIR="${OUT_DIR:-runs/mednext_utsw_causal_proxy}"
EPOCHS="${CAUSAL_EPOCHS:-${EPOCHS:-20}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CACHE_DIR="${CACHE_DIR:-runs/cache/utsw_mednext_causal_v64}"
DEFAULT_METADATA_PATH="${DEFAULT_METADATA_PATH:-data/brats/UTSW_Glioma_Metadata-2-1.tsv}"
METADATA_PATH="${METADATA_PATH:-$DEFAULT_METADATA_PATH}"

if [[ ! -f "$BASELINE_CHECKPOINT" ]]; then
  echo "Missing baseline checkpoint: $BASELINE_CHECKPOINT" >&2
  echo "Run scripts/train_mednext_utsw_proxy_transport_baseline.sh first, or set BASELINE_CHECKPOINT." >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN" baselines/mednext/train_causal_utsw.py
  --baseline-checkpoint "$BASELINE_CHECKPOINT"
  --data-root "${DATA_ROOT:-data/brats/PKG - UTSW-Glioma/UTSW-Glioma}"
  --output-dir "$OUT_DIR"
  --model-id "${MODEL_ID:-S}"
  --kernel-size "${KERNEL_SIZE:-3}"
  --volume-size "${VOLUME_SIZE:-64}"
  --latent-dim "${LATENT_DIM:-128}"
  --epochs "$EPOCHS"
  --batch-size "${BATCH_SIZE:-1}"
  --lr "${LR:-0.00005}"
  --backbone-lr "${BACKBONE_LR:-0.00001}"
  --causal-lr "${CAUSAL_LR:-0.0002}"
  --weight-decay "${WEIGHT_DECAY:-0.0001}"
  --threshold "${THRESHOLD:-0.5}"
  --crop-margin "${CROP_MARGIN:-8}"
  --disk-cache-dir "$CACHE_DIR"
  --context-bank-size "${CONTEXT_BANK_SIZE:-64}"
  --context-bank-sampling "${CONTEXT_BANK_SAMPLING:-farthest}"
  --adjustment-contexts "${ADJUSTMENT_CONTEXTS:-2}"
  --context-bank-refresh-epochs "${CONTEXT_BANK_REFRESH_EPOCHS:-2}"
  --freeze-backbone-epochs "${FREEZE_BACKBONE_EPOCHS:-4}"
  --max-context-bank-batches "${MAX_CONTEXT_BANK_BATCHES:-64}"
  --lambda-seg "${LAMBDA_SEG:-1.0}"
  --lambda-region-loss "${LAMBDA_REGION_LOSS:-0.10}"
  --lambda-adjustment "${LAMBDA_ADJUSTMENT:-0.30}"
  --lambda-context-stability "${LAMBDA_CONTEXT_STABILITY:-0.02}"
  --lambda-context-from-disease-adversary "${LAMBDA_CONTEXT_FROM_DISEASE_ADVERSARY:-0.02}"
  --lambda-disease-from-context-adversary "${LAMBDA_DISEASE_FROM_CONTEXT_ADVERSARY:-0.02}"
  --lambda-region-volume-proxy "${LAMBDA_REGION_VOLUME_PROXY:-0.05}"
  --lambda-region-from-context-adversary "${LAMBDA_REGION_FROM_CONTEXT_ADVERSARY:-0.02}"
  --lambda-orthogonal "${LAMBDA_ORTHOGONAL:-0.01}"
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
if [[ -n "${SPLITS_JSON:-}" ]]; then
  cmd+=(--splits-json "$SPLITS_JSON")
fi
if [[ -n "${LIMIT_CASES:-}" ]]; then
  cmd+=(--limit-cases "$LIMIT_CASES")
fi
if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  cmd+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "${MAX_VAL_BATCHES:-}" ]]; then
  cmd+=(--max-val-batches "$MAX_VAL_BATCHES")
fi
if [[ -n "${ADJUSTMENT_CONTEXT_SELECTION:-}" ]]; then
  cmd+=(--adjustment-context-selection "$ADJUSTMENT_CONTEXT_SELECTION")
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
if [[ -n "${CHECKPOINT_CALIBRATION_THRESHOLDS:-}" ]]; then
  cmd+=(--checkpoint-calibration-thresholds "$CHECKPOINT_CALIBRATION_THRESHOLDS")
fi
if [[ -n "${CHECKPOINT_CALIBRATION_OBJECTIVE:-}" ]]; then
  cmd+=(--checkpoint-calibration-objective "$CHECKPOINT_CALIBRATION_OBJECTIVE")
fi

"${cmd[@]}"
