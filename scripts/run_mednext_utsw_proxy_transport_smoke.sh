#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="${BASELINE_DIR:-runs/_tmp_mednext_utsw_baseline_fast_smoke}"
CAUSAL_DIR="${CAUSAL_DIR:-runs/_tmp_mednext_utsw_causal_fast_smoke}"
REFERENCE_BASELINE_CHECKPOINT="${REFERENCE_BASELINE_CHECKPOINT:-runs/mednext_utsw_s_k3/best.pt}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-}"
LIMIT_CASES="${LIMIT_CASES:-8}"
FORCE="${FORCE:-0}"

if [[ -z "$BASELINE_CHECKPOINT" && -f "$REFERENCE_BASELINE_CHECKPOINT" ]]; then
  BASELINE_CHECKPOINT="$REFERENCE_BASELINE_CHECKPOINT"
fi

if [[ -n "$BASELINE_CHECKPOINT" ]]; then
  echo "Using existing proxy-transport baseline checkpoint: $BASELINE_CHECKPOINT"
elif [[ "${ALLOW_SCRATCH_BASELINE:-0}" != "1" ]]; then
  echo "Missing original trained baseline: $REFERENCE_BASELINE_CHECKPOINT" >&2
  echo "Set BASELINE_CHECKPOINT to a trained UTSW MedNeXt checkpoint, or set ALLOW_SCRATCH_BASELINE=1 for a code-only smoke." >&2
  exit 1
elif [[ "$FORCE" == "1" || ! -f "$BASELINE_DIR/best.pt" ]]; then
  echo "Reference baseline not found at $REFERENCE_BASELINE_CHECKPOINT; training a scratch smoke baseline." >&2
  echo "This scratch smoke path is runnable, but it is not expected to reproduce the old 0.908 UTSW result." >&2
  OUT_DIR="$BASELINE_DIR" BASELINE_EPOCHS=1 LIMIT_CASES="$LIMIT_CASES" \
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-1}" MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-1}" \
    bash scripts/train_mednext_utsw_proxy_transport_baseline.sh
  BASELINE_CHECKPOINT="$BASELINE_DIR/best.pt"
else
  echo "Using existing smoke baseline checkpoint: $BASELINE_DIR/best.pt"
  BASELINE_CHECKPOINT="$BASELINE_DIR/best.pt"
fi

if [[ "$FORCE" == "1" || ! -f "$CAUSAL_DIR/best.pt" ]]; then
  BASELINE_CHECKPOINT="$BASELINE_CHECKPOINT" OUT_DIR="$CAUSAL_DIR" CAUSAL_EPOCHS=1 LIMIT_CASES="$LIMIT_CASES" \
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-1}" MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-1}" \
  MAX_CONTEXT_BANK_BATCHES="${MAX_CONTEXT_BANK_BATCHES:-2}" CONTEXT_BANK_SIZE="${CONTEXT_BANK_SIZE:-4}" \
    bash scripts/train_mednext_utsw_proxy_transport_causal.sh
else
  echo "Using existing smoke causal checkpoint: $CAUSAL_DIR/best.pt"
fi

CHECKPOINT="$CAUSAL_DIR/best.pt" OUT_JSON="$CAUSAL_DIR/utsw_val_causal_metrics.json" \
MAX_BATCHES="${EVAL_MAX_BATCHES:-1}" MAX_CONTEXT_BANK_BATCHES="${EVAL_MAX_CONTEXT_BANK_BATCHES:-2}" \
CCT_CONTEXTS="${CCT_CONTEXTS:-2}" CONTEXT_BANK_SIZE="${CONTEXT_BANK_SIZE:-4}" \
  bash scripts/evaluate_mednext_utsw_proxy_transport_causal.sh
