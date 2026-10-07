#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="${BASELINE_DIR:-runs/mednext_brats_h5_s_k3_paper64}"
CAUSAL_DIR="${CAUSAL_DIR:-runs/mednext_brats_h5_causal_s_k3_paper64}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-50}"
CAUSAL_EPOCHS="${CAUSAL_EPOCHS:-12}"
FORCE="${FORCE:-0}"

if [[ "$FORCE" == "1" || ! -f "$BASELINE_DIR/best.pt" ]]; then
  OUT_DIR="$BASELINE_DIR" BASELINE_EPOCHS="$BASELINE_EPOCHS" bash scripts/train_mednext_brats_paper64_baseline.sh
else
  echo "Using existing baseline checkpoint: $BASELINE_DIR/best.pt"
fi

CHECKPOINT="$BASELINE_DIR/best.pt" \
OUT_JSON="$BASELINE_DIR/brats_val_metrics.json" \
  bash scripts/evaluate_mednext_brats_paper64_baseline.sh

if [[ "$FORCE" == "1" || ! -f "$CAUSAL_DIR/best.pt" ]]; then
  BASELINE_DIR="$BASELINE_DIR" OUT_DIR="$CAUSAL_DIR" CAUSAL_EPOCHS="$CAUSAL_EPOCHS" \
    bash scripts/train_mednext_brats_paper64_causal.sh
else
  echo "Using existing causal checkpoint: $CAUSAL_DIR/best.pt"
fi

CHECKPOINT="$CAUSAL_DIR/best.pt" \
OUT_JSON="$CAUSAL_DIR/brats_val_causal_metrics.json" \
  bash scripts/evaluate_mednext_brats_paper64_causal.sh
