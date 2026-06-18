#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x .venv312_restore/bin/python ]]; then
    PYTHON_BIN=.venv312_restore/bin/python
  elif [[ -x .venv312/bin/python ]]; then
    PYTHON_BIN=.venv312/bin/python
  elif [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=.venv/bin/python
  else
    PYTHON_BIN=python
  fi
fi

export PYTHONPATH=".:src${PYTHONPATH:+:$PYTHONPATH}"

ACTION="${1:-list}"
OOD_DIRECTION="${2:-utsw-to-brats}"
EXECUTE="${EXECUTE:-0}"
CALIBRATION_THRESHOLDS="${CALIBRATION_THRESHOLDS:-0.3,0.4,0.5,0.6,0.7,0.8,0.9}"
CALIBRATION_OBJECTIVE="${CALIBRATION_OBJECTIVE:-tc_et_min}"
THRESHOLD="${THRESHOLD:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"

usage() {
  cat <<'USAGE'
Run MedNeXt source-only OOD baseline evaluations.

Usage:
  bash baselines/run_mednext_ood_baselines.sh list
  bash baselines/run_mednext_ood_baselines.sh mednext utsw-to-brats
  bash baselines/run_mednext_ood_baselines.sh mednext brats-to-utsw

Dry-run is the default. Set EXECUTE=1 to run.

Common env:
  PYTHON_BIN=.venv312_restore/bin/python
  EXECUTE=1
  MAX_VOLUMES=4          # UTSW -> BraTS target volume limit
  MAX_CASES=4            # BraTS -> UTSW target case limit; 0 means all split cases
  CHECKPOINT=...         # override source checkpoint
  OUT_JSON=...           # override output json
  CALIBRATION_THRESHOLDS=0.3,0.4,0.5,0.6,0.7,0.8,0.9
  CALIBRATION_OBJECTIVE=tc_et_min

Defaults:
  utsw-to-brats checkpoint: runs/mednext_utsw_s_k3/best.pt
  brats-to-utsw checkpoint: runs/mednext_brats_h5_s_k3_paper64/best.pt
USAGE
}

print_or_exec() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$EXECUTE" == "1" ]]; then
    "$@"
  fi
}

utsw_case_ids() {
  local splits_json="${UTSW_SPLITS_JSON:-runs/mednext_utsw_s_k3/splits.json}"
  local split="${UTSW_SPLIT:-val}"
  local max_cases="${MAX_CASES:-4}"
  "$PYTHON_BIN" - "$splits_json" "$split" "$max_cases" <<'PY'
import json
import sys
from pathlib import Path

splits_path = Path(sys.argv[1])
split = sys.argv[2]
max_cases = int(sys.argv[3])
if not splits_path.exists():
    raise SystemExit(f"Missing UTSW split file: {splits_path}. Set TARGET_CASE_IDS or UTSW_SPLITS_JSON.")
splits = json.loads(splits_path.read_text(encoding="utf-8"))
case_ids = list(splits.get(split, []))
if not case_ids:
    raise SystemExit(f"No UTSW case ids found for split {split!r} in {splits_path}")
if max_cases > 0:
    case_ids = case_ids[:max_cases]
print(",".join(case_ids))
PY
}

run_mednext_utsw_to_brats() {
  local max_volumes="${MAX_VOLUMES:-4}"
  local checkpoint="${CHECKPOINT:-runs/mednext_utsw_s_k3/best.pt}"
  local out_json="${OUT_JSON:-runs/baseline_compare/mednext_utsw_to_brats_val${max_volumes}.json}"
  local split_name="${SPLIT_NAME:-ood_mednext_utsw_to_brats_val${max_volumes}}"

  print_or_exec env \
    PYTHON_BIN="$PYTHON_BIN" \
    CHECKPOINT="$checkpoint" \
    OUT_JSON="$out_json" \
    MAX_VOLUMES="$max_volumes" \
    CALIBRATION_THRESHOLDS="$CALIBRATION_THRESHOLDS" \
    CALIBRATION_OBJECTIVE="$CALIBRATION_OBJECTIVE" \
    THRESHOLD="$THRESHOLD" \
    NUM_WORKERS="$NUM_WORKERS" \
    SPLIT_NAME="$split_name" \
    bash scripts/evaluate_mednext_brats_paper64_baseline.sh
}

run_mednext_brats_to_utsw() {
  local checkpoint="${CHECKPOINT:-runs/mednext_brats_h5_s_k3_paper64/best.pt}"
  local max_cases="${MAX_CASES:-4}"
  local out_suffix="$max_cases"
  if [[ "$max_cases" == "0" ]]; then
    out_suffix="full"
  fi
  local out_json="${OUT_JSON:-runs/baseline_compare/mednext_brats_to_utsw_val${out_suffix}.json}"
  local case_ids="${TARGET_CASE_IDS:-}"
  if [[ -z "$case_ids" ]]; then
    case_ids="$(utsw_case_ids)"
  fi

  local cmd=(
    "$PYTHON_BIN" baselines/mednext/evaluate_utsw.py
    --checkpoint "$checkpoint"
    --split "${UTSW_SPLIT:-val}"
    --case-ids "$case_ids"
    --data-root "${UTSW_DATA_ROOT:-data/brats/PKG - UTSW-Glioma/UTSW-Glioma}"
    --output-json "$out_json"
    --threshold "$THRESHOLD"
    --calibration-thresholds "$CALIBRATION_THRESHOLDS"
    --calibration-objective "$CALIBRATION_OBJECTIVE"
    --num-workers "$NUM_WORKERS"
  )
  if [[ -n "${REGION_THRESHOLDS:-}" ]]; then
    cmd+=(--region-thresholds "$REGION_THRESHOLDS")
  fi
  if [[ "${STRUCTURAL_PRIOR:-0}" == "1" ]]; then
    cmd+=(--structural-prior)
    if [[ -n "${STRUCTURAL_THRESHOLD:-}" ]]; then
      cmd+=(--structural-threshold "$STRUCTURAL_THRESHOLD")
    fi
    if [[ -n "${STRUCTURAL_MIN_COMPONENT_SIZE:-}" ]]; then
      cmd+=(--structural-min-component-size "$STRUCTURAL_MIN_COMPONENT_SIZE")
    fi
  fi
  print_or_exec "${cmd[@]}"
}

if [[ "$ACTION" == "list" || "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$ACTION" != "mednext" ]]; then
  echo "Unsupported action/model: $ACTION" >&2
  usage >&2
  exit 2
fi

case "$OOD_DIRECTION" in
  utsw-to-brats)
    run_mednext_utsw_to_brats
    ;;
  brats-to-utsw)
    run_mednext_brats_to_utsw
    ;;
  *)
    echo "Unsupported OOD direction: $OOD_DIRECTION" >&2
    usage >&2
    exit 2
    ;;
esac
