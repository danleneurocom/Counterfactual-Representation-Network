# Causal MedNeXt for OOD Brain Tumor Segmentation

This repository contains the code used for our current causal MedNeXt
out-of-distribution brain tumor segmentation experiments. The main model is a
3D MedNeXt segmenter augmented with a structural causal mechanism for
context-aware segmentation, TC/ET-aware calibration, and anatomical
region-plausibility constraints.

The current strongest verified OOD smoke result is UTSW -> BraTS:

```text
Mean Dice 0.792 | WT 0.852 | TC 0.820 | ET 0.703
```

The corresponding evaluation artifact is:

```text
runs/_ood_causal_adapt_brats_v5_et_precision_e2/brats_val_structural_min32_metrics.json
```

Large checkpoints, run folders, and medical imaging data are intentionally not
tracked in git. Share checkpoints separately as artifacts.

## Repository Map

```text
src/causal_mednext/       Canonical Causal MedNeXt model package
src/causal_mednext/backbone.py
                           3D MedNeXt encoder-decoder backbone
src/causal_mednext/causal_model.py
                           SCM-style disease/context model and adjustment logic
src/causal_mednext/mechanisms/
                           Calibration, structural prior, hierarchy, and veto mechanisms
baselines/mednext/        Train/eval wrappers and compatibility imports for scripts
baselines/segformer3d/    Local SegFormer3D baseline and shared UTSW/BraTS loaders
baselines/README.md       Baseline comparison and OOD baseline entry points
scripts/                  Reproducible train/eval wrappers for the MedNeXt OOD path
src/crn/metrics.py        BraTS metrics and structural-prior evaluation utilities
tests/                    Focused regression tests for metrics and MedNeXt plumbing
docs/                     Reproduction notes and public-release checklist
```

The legacy CRN prototype code is kept only where it supports the current
MedNeXt path. The public mainline model is `causal_mednext`; the
`baselines/mednext` package keeps the original experiment commands stable.

## Setup

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the PyTorch build appropriate for your machine if needed:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use `PYTHONPATH=.:src` for direct script execution.

## Data Layout

The scripts expect local datasets in these locations:

```text
data/brats/brats_train.csv
data/brats/brats_val.csv
data/brats/archive/BraTS2020_training_data/content/data/
data/brats/PKG - UTSW-Glioma/UTSW-Glioma/
data/brats/UTSW_Glioma_Metadata-2-1.tsv
```

These files are ignored by git because they are large and/or dataset controlled.

If you need to regenerate the BraTS HDF5 CSVs:

```bash
python scripts/prepare_brats_metadata.py \
  --metadata "BraTS20 Training Metadata.csv" \
  --data-root /path/to/BraTS2020_training_data/content/data \
  --output-dir data/brats \
  --require-files
```
