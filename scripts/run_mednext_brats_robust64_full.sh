#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="${BASELINE_DIR:-runs/mednext_brats_h5_s_k3_robust64}"
CAUSAL_DIR="${CAUSAL_DIR:-runs/mednext_brats_h5_causal_s_k3_robust64}"
CACHE_DIR="${CACHE_DIR:-runs/cache/brats_h5_robust64_v64}"

BASELINE_DIR="$BASELINE_DIR" \
CAUSAL_DIR="$CAUSAL_DIR" \
CACHE_DIR="$CACHE_DIR" \
  bash scripts/run_mednext_brats_paper64_full.sh
