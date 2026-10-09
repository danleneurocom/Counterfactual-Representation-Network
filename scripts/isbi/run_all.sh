#!/usr/bin/env bash
# Full ISBI experiment matrix with one command.
#
#   1. train   : {baseline, styleaug, trace} x SEEDS  +  ablations x ABL_SEEDS, for each source dataset
#   2. calib   : thresholds + ET prior on the SOURCE val split of the same run
#   3. eval    : ID test (source test split) and OOD test (other dataset's test split),
#                with TTA, CCT (source bank from the checkpoint) and seed-ensemble signals
#   4. audit   : failure-detection metrics of every audit signal, per source->target and method
#   5. tables  : mean +- std over seeds, Wilcoxon vs baseline (final = calibrated + prior,
#                raw = 0.5 thresholds without prior -> structural-prior ablation for free)
#
# Real run (GPU):
#   DATA_UTSW=data/processed/utsw_128 DATA_BRATS=data/processed/brats_128 \
#   SPLIT_UTSW=splits/utsw.json SPLIT_BRATS=splits/brats.json bash scripts/isbi/run_all.sh
#
# Everything is resumable: finished steps are skipped (SKIP_EXISTING=1).
set -euo pipefail

PY="${PYTHON_BIN:-python}"
export PYTHONPATH=".:src${PYTHONPATH:+:$PYTHONPATH}"

DATA_UTSW="${DATA_UTSW:-data/processed/utsw_128}"
DATA_BRATS="${DATA_BRATS:-data/processed/brats_128}"
SPLIT_UTSW="${SPLIT_UTSW:-splits/utsw.json}"
SPLIT_BRATS="${SPLIT_BRATS:-splits/brats.json}"
RUNS="${RUNS:-runs/isbi}"
RESULTS="${RESULTS:-results/isbi}"
SOURCES="${SOURCES:-utsw brats}"
SEEDS="${SEEDS:-0 1 2}"
ABL_SEEDS="${ABL_SEEDS-0}"
EPOCHS="${EPOCHS:-150}"
WORKERS="${WORKERS:-4}"
AMP="${AMP:-1}"
CCT_K="${CCT_K:-4}"
SAVE_MAPS="${SAVE_MAPS:-6}"   # qualitative maps per eval (~20-40 MB each)
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"   # e.g. "--base-channels 8 --patch-size 48" for smoke tests
STEPS="${STEPS:-train calib eval audit tables}"

# method name -> extra training flags
declare -A METHOD_ARGS=(
  [baseline]="--mode baseline"
  [styleaug]="--mode styleaug"
  [trace]="--mode trace"
)
declare -A ABLATION_ARGS=(
  [trace_nocct]="--mode trace --no-cct-loss"
  [trace_nostab]="--mode trace --no-stability"
  [trace_noproxy]="--mode trace --no-proxies"
)
MAIN_METHODS="${MAIN_METHODS-baseline styleaug trace}"  # set to "" to skip
ABLATIONS="${ABLATIONS-trace_nocct trace_nostab trace_noproxy}"  # set to "" to skip

amp_flag=""; [[ "$AMP" == "1" ]] && amp_flag="--amp"
data_of()  { [[ "$1" == "utsw" ]] && echo "$DATA_UTSW"  || echo "$DATA_BRATS"; }
split_of() { [[ "$1" == "utsw" ]] && echo "$SPLIT_UTSW" || echo "$SPLIT_BRATS"; }
other_of() { [[ "$1" == "utsw" ]] && echo "brats" || echo "utsw"; }
want() { [[ " $STEPS " == *" $1 "* ]]; }
have() { [[ -f "$(split_of "$1")" && -d "$(data_of "$1")" ]]; }  # dataset preprocessed + split exists

runs_for() {  # prints "method seed flags" lines
  for m in $MAIN_METHODS; do for s in $SEEDS; do echo "$m|$s|${METHOD_ARGS[$m]}"; done; done
  for m in $ABLATIONS; do for s in $ABL_SEEDS; do echo "$m|$s|${ABLATION_ARGS[$m]}"; done; done
}

for src in $SOURCES; do
  tgt="$(other_of "$src")"
  if ! have "$src"; then
    echo "!! source '$src' skipped: missing $(data_of "$src") or $(split_of "$src") (run preprocess + splits first)"
    continue
  fi
  while IFS='|' read -r method seed flags; do
    out="$RUNS/$src/${method}_s${seed}"
    if want train; then
      if [[ "$SKIP_EXISTING" == "1" && -f "$out/summary.json" ]]; then
        echo "[skip train] $out"
      else
        echo "[train] $out"
        # shellcheck disable=SC2086
        "$PY" -m trace_seg3d.train --data-dir "$(data_of "$src")" --splits "$(split_of "$src")" \
          --out "$out" --seed "$seed" --epochs "$EPOCHS" --workers "$WORKERS" $amp_flag $flags $EXTRA_TRAIN_ARGS
      fi
    fi
    if want calib; then
      if [[ "$SKIP_EXISTING" == "1" && -f "$out/calib.json" ]]; then echo "[skip calib] $out"; else
        "$PY" -m trace_seg3d.calibrate --ckpt "$out/best.pt" --data-dir "$(data_of "$src")" --splits "$(split_of "$src")" --workers "$WORKERS" $amp_flag
      fi
    fi
  done < <(runs_for)

  if want eval; then
    while IFS='|' read -r method seed flags; do
      out="$RUNS/$src/${method}_s${seed}"
      ens=()
      for s in $SEEDS; do [[ "$s" != "$seed" && -f "$RUNS/$src/${method}_s${s}/best.pt" ]] && ens+=("$RUNS/$src/${method}_s${s}/best.pt"); done
      for target in "$src" "$tgt"; do
        if ! have "$target"; then echo "[skip eval] target '$target' not available"; continue; fi
        ev="$out/eval_${target}_test"
        if [[ "$SKIP_EXISTING" == "1" && -f "$ev/summary.json" ]]; then echo "[skip eval] $ev"; continue; fi
        echo "[eval] $ev"
        "$PY" -m trace_seg3d.evaluate --ckpt "$out/best.pt" --calib "$out/calib.json" \
          --data-dir "$(data_of "$target")" --splits "$(split_of "$target")" --split test \
          --cct-k "$CCT_K" --tta --save-maps "$SAVE_MAPS" --workers "$WORKERS" $amp_flag --out "$ev" \
          ${ens:+--ensemble-ckpts "${ens[@]}"}
      done
    done < <(runs_for)
  fi

  if want audit; then
    mkdir -p "$RESULTS"
    for method in $MAIN_METHODS $ABLATIONS; do
      for target in "$src" "$tgt"; do
        shopt -s nullglob
        csvs=("$RUNS/$src/${method}"_s*/eval_"${target}"_test/per_case.csv)
        shopt -u nullglob
        [[ ${#csvs[@]} -eq 0 ]] && continue
        "$PY" -m trace_seg3d.audit "${csvs[@]}" --out "$RESULTS/audit_${src}_to_${target}_${method}.md" >/dev/null
        "$PY" -m trace_seg3d.audit "${csvs[@]}" --fail-quantile 0.2 --out "$RESULTS/audit_${src}_to_${target}_${method}_q20.md" >/dev/null
      done
    done
  fi
done

if want tables; then
  "$PY" -m trace_seg3d.summarize --root "$RUNS" --prefix final_ --out "$RESULTS/final"
  "$PY" -m trace_seg3d.summarize --root "$RUNS" --prefix raw_ --out "$RESULTS/raw_no_prior"
  "$PY" -m trace_seg3d.summarize --root "$RUNS" --prefix cct_final_ --out "$RESULTS/cct_consensus" || true
  echo "tables -> $RESULTS"
fi
