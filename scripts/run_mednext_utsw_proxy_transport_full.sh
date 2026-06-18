#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="${BASELINE_DIR:-runs/mednext_utsw_s_k3}"
CAUSAL_DIR="${CAUSAL_DIR:-runs/mednext_utsw_causal_proxy}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-100}"
CAUSAL_EPOCHS="${CAUSAL_EPOCHS:-20}"
FORCE="${FORCE:-0}"

if [[ "$FORCE" == "1" || ! -f "$BASELINE_DIR/best.pt" ]]; then
  OUT_DIR="$BASELINE_DIR" BASELINE_EPOCHS="$BASELINE_EPOCHS" \
    bash scripts/train_mednext_utsw_proxy_transport_baseline.sh
else
  echo "Using existing UTSW baseline checkpoint: $BASELINE_DIR/best.pt"
fi

if [[ "$FORCE" == "1" || ! -f "$CAUSAL_DIR/best.pt" ]]; then
  BASELINE_DIR="$BASELINE_DIR" OUT_DIR="$CAUSAL_DIR" CAUSAL_EPOCHS="$CAUSAL_EPOCHS" \
    bash scripts/train_mednext_utsw_proxy_transport_causal.sh
else
  echo "Using existing UTSW causal checkpoint: $CAUSAL_DIR/best.pt"
fi

CHECKPOINT="$CAUSAL_DIR/best.pt" OUT_JSON="$CAUSAL_DIR/utsw_val_causal_metrics.json" \
  bash scripts/evaluate_mednext_utsw_proxy_transport_causal.sh
