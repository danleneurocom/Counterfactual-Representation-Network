#!/usr/bin/env bash
# End-to-end smoke test of the ISBI pipeline on tiny synthetic data (CPU is fine).
# Numbers are meaningless; this only proves every step runs and produces the tables.
set -euo pipefail
PY="${PYTHON_BIN:-python}"
export PYTHONPATH=".:src${PYTHONPATH:+:$PYTHONPATH}"
S="${SMOKE_DIR:-runs/smoke_synthetic}"
rm -rf "$S"
"$PY" scripts/isbi/make_synthetic_data.py --out "$S/raw" --n-utsw 24 --n-brats 24
"$PY" -m trace_seg3d.preprocess utsw --root "$S/raw/PKG - UTSW-Glioma/UTSW-Glioma" \
  --metadata "$S/raw/UTSW_Glioma_Metadata-2-1.tsv" --out "$S/proc/utsw" --size 48 --workers 2
"$PY" -m trace_seg3d.preprocess brats-nifti --root "$S/raw/MICCAI_BraTS2020_TrainingData" --out "$S/proc/brats" --size 48 --workers 2
"$PY" -m trace_seg3d.check_data "$S/proc/utsw" "$S/proc/brats" --png "$S/check.png"
"$PY" -m trace_seg3d.splits --index "$S/proc/utsw/index.csv" --out "$S/splits/utsw.json"
"$PY" -m trace_seg3d.splits --index "$S/proc/brats/index.csv" --out "$S/splits/brats.json"
PYTHON_BIN="$PY" DATA_UTSW="$S/proc/utsw" DATA_BRATS="$S/proc/brats" \
SPLIT_UTSW="$S/splits/utsw.json" SPLIT_BRATS="$S/splits/brats.json" \
RUNS="$S/runs" RESULTS="$S/results" SEEDS="0 1" ABL_SEEDS="0" EPOCHS="${EPOCHS:-10}" WORKERS=0 AMP=0 \
EXTRA_TRAIN_ARGS="--base-channels 8 --patch-size 48 --batch-size 2 --cct-start-epoch 2 --val-every 2 --lr 3e-3 --warmup-epochs 1" \
bash scripts/isbi/run_all.sh
echo "smoke OK -> $S/results"
