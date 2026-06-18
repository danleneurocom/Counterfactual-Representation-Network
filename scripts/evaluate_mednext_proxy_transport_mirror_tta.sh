#!/usr/bin/env bash
set -euo pipefail

MIRROR_TTA_AXES="${MIRROR_TTA_AXES:-d,h,w}"

if [[ -n "${BRATS_CHECKPOINT:-}" ]]; then
  CHECKPOINT="$BRATS_CHECKPOINT" \
  OUT_JSON="${BRATS_OUT_JSON:-${BRATS_CHECKPOINT%.pt}_mirror_tta.json}" \
  MIRROR_TTA_AXES="$MIRROR_TTA_AXES" \
    bash scripts/evaluate_mednext_brats_paper64_causal.sh
fi

if [[ -n "${UTSW_CHECKPOINT:-}" ]]; then
  CHECKPOINT="$UTSW_CHECKPOINT" \
  OUT_JSON="${UTSW_OUT_JSON:-${UTSW_CHECKPOINT%.pt}_mirror_tta.json}" \
  MIRROR_TTA_AXES="$MIRROR_TTA_AXES" \
    bash scripts/evaluate_mednext_utsw_proxy_transport_causal.sh
fi

if [[ -z "${BRATS_CHECKPOINT:-}" && -z "${UTSW_CHECKPOINT:-}" ]]; then
  echo "Set BRATS_CHECKPOINT and/or UTSW_CHECKPOINT to evaluate mirror TTA." >&2
  exit 1
fi
