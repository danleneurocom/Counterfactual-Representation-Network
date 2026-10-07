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

BASELINE_DIR="${BASELINE_DIR:-runs/mednext_brats_h5_s_k3_paper64}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$BASELINE_DIR/best.pt}"
OUT_DIR="${OUT_DIR:-runs/mednext_brats_h5_causal_s_k3_paper64}"
EPOCHS="${CAUSAL_EPOCHS:-${EPOCHS:-12}}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CACHE_DIR="${CACHE_DIR:-runs/cache/brats_h5_paper64_v64}"

if [[ ! -f "$BASELINE_CHECKPOINT" ]]; then
  echo "Missing baseline checkpoint: $BASELINE_CHECKPOINT" >&2
  echo "Run scripts/train_mednext_brats_paper64_baseline.sh first, or set BASELINE_CHECKPOINT." >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN" baselines/mednext/train_causal_brats_h5.py
  --baseline-checkpoint "$BASELINE_CHECKPOINT"
  --train-csv "${TRAIN_CSV:-data/brats/brats_train.csv}"
  --val-csv "${VAL_CSV:-data/brats/brats_val.csv}"
  --data-root "${DATA_ROOT:-data/brats/archive/BraTS2020_training_data/content/data}"
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
  --lambda-context-proxy "${LAMBDA_CONTEXT_PROXY:-0.03}"
  --lambda-disease-proxy "${LAMBDA_DISEASE_PROXY:-0.05}"
  --lambda-annotation-proxy "${LAMBDA_ANNOTATION_PROXY:-0.01}"
  --lambda-context-from-disease-adversary "${LAMBDA_CONTEXT_FROM_DISEASE_ADVERSARY:-0.02}"
  --lambda-disease-from-context-adversary "${LAMBDA_DISEASE_FROM_CONTEXT_ADVERSARY:-0.02}"
  --lambda-region-volume-proxy "${LAMBDA_REGION_VOLUME_PROXY:-0.05}"
  --lambda-region-from-context-adversary "${LAMBDA_REGION_FROM_CONTEXT_ADVERSARY:-0.02}"
  --lambda-sdd-context-teacher "${LAMBDA_SDD_CONTEXT_TEACHER:-0.0}"
  --lambda-sdd-region-teacher "${LAMBDA_SDD_REGION_TEACHER:-0.0}"
  --lambda-sdd-context-distill "${LAMBDA_SDD_CONTEXT_DISTILL:-0.0}"
  --lambda-sdd-region-distill "${LAMBDA_SDD_REGION_DISTILL:-0.0}"
  --lambda-sdd-treatment "${LAMBDA_SDD_TREATMENT:-0.0}"
  --lambda-sdd-treatment-disentangle "${LAMBDA_SDD_TREATMENT_DISENTANGLE:-0.0}"
  --lambda-sdd-outcome "${LAMBDA_SDD_OUTCOME:-0.0}"
  --lambda-sdd-outcome-disentangle "${LAMBDA_SDD_OUTCOME_DISENTANGLE:-0.0}"
  --lambda-sdd-imbalance "${LAMBDA_SDD_IMBALANCE:-0.0}"
  --lambda-cite-contrastive "${LAMBDA_CITE_CONTRASTIVE:-0.0}"
  --lambda-orthogonal "${LAMBDA_ORTHOGONAL:-0.01}"
  --num-workers "$NUM_WORKERS"
)

if [[ "${USE_PSEUDO_PROXIES:-1}" == "1" ]]; then
  cmd+=(--use-pseudo-proxies)
fi
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  cmd+=(--init-checkpoint "$INIT_CHECKPOINT")
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
if [[ -n "${LIMIT_TRAIN_VOLUMES:-}" ]]; then
  cmd+=(--limit-train-volumes "$LIMIT_TRAIN_VOLUMES")
fi
if [[ -n "${LIMIT_VAL_VOLUMES:-}" ]]; then
  cmd+=(--limit-val-volumes "$LIMIT_VAL_VOLUMES")
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
if [[ -n "${CHECKPOINT_CALIBRATION_THRESHOLDS:-}" ]]; then
  cmd+=(--checkpoint-calibration-thresholds "$CHECKPOINT_CALIBRATION_THRESHOLDS")
fi
if [[ -n "${CHECKPOINT_CALIBRATION_OBJECTIVE:-}" ]]; then
  cmd+=(--checkpoint-calibration-objective "$CHECKPOINT_CALIBRATION_OBJECTIVE")
fi

"${cmd[@]}"
