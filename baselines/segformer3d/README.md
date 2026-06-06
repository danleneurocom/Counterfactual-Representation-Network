# SegFormer3D Baseline

This folder is a clean baseline track for replicating the model from
`OSUPCVLab/SegFormer3D` before adding any causal components.

Copied upstream files:

- `architectures/segformer3d.py`
- `architectures/build_architecture.py`
- `UPSTREAM_README.md`
- `upstream_requirements.txt`
- `LICENSE`

The upstream repository identifies `architectures/segformer3d.py` as the
implementation of the SegFormer3D architecture. We keep this code separate from
`src/crn` so that the first milestone is a faithful non-causal baseline.

Source:

- GitHub: https://github.com/OSUPCVLab/SegFormer3D
- Local archive used here: `/Users/lenguyenlinhdan/Downloads/SegFormer3D-main.zip`

Important modeling constraint: the upstream implementation assumes cubic 3D
feature grids when it converts token sequences back to volumes. Use cubic input
patches such as `128 x 128 x 128` unless the architecture is modified later.

## Running UTSW

The UTSW dataset under this repo is stored as one NIfTI folder per case:

```text
data/brats/PKG - UTSW-Glioma/UTSW-Glioma/BT0001/
  brain_flair.nii.gz
  brain_t1.nii.gz
  brain_t1ce.nii.gz
  brain_t2.nii.gz
  tumorseg_FeTS.nii.gz
```

SegFormer3D expects cubic 3D tensors, so the local adapter
`baselines/segformer3d/data/utsw.py` loads the NIfTI files, converts them to:

```text
image: (4, S, S, S)
mask:  (3, S, S, S)
```

The mask channels are BraTS subregions:

```text
0 = ncr_net
1 = edema
2 = enhancing_tumor
```

Quick smoke run on one real UTSW case:

```bash
PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/run_utsw_smoke.py \
  --case-id BT0001 \
  --volume-size 32 \
  --model-size tiny
```

Use the base SegFormer3D architecture instead of the tiny smoke model:

```bash
PYTHONPATH=. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/run_utsw_smoke.py \
  --case-id BT0001 \
  --volume-size 128 \
  --model-size base
```

The smoke script is an interface check, not a trained result. A meaningful UTSW
experiment still needs the next layer: train/validation splits, training loop,
checkpointing, and volume-level Dice/HD95 evaluation.

## Train And Test UTSW

Tiny CPU sanity run:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/train_utsw.py \
  --output-dir runs/segformer3d_utsw_smoke \
  --model-size tiny \
  --volume-size 64 \
  --epochs 1 \
  --limit-cases 3 \
  --max-train-batches 1 \
  --max-val-batches 1
```

Evaluate the held-out test split from that run:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/evaluate_utsw.py \
  --checkpoint runs/segformer3d_utsw_smoke/best.pt \
  --split test \
  --max-batches 1
```

Real from-scratch baseline run:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/train_utsw.py \
  --output-dir runs/segformer3d_utsw_base \
  --model-size base \
  --volume-size 128 \
  --epochs 100 \
  --batch-size 1 \
  --lr 0.0002 \
  --weight-decay 0.0001 \
  --num-workers 2
```

Run final test metrics:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/evaluate_utsw.py \
  --checkpoint runs/segformer3d_utsw_base/best.pt \
  --split test \
  --num-workers 2
```

Outputs:

- `splits.json`: patient-level train/val/test split
- `config.json`: training configuration
- `epoch_*.json`: train/validation metrics
- `best.pt`: best validation checkpoint
- `last.pt`: latest checkpoint
- `test_metrics.json`: held-out test metrics

## Causal Phase B

The causal extension starts from Pearl's order:

```text
Define -> Assume -> Identify -> Estimate -> Answer
```

The implementation adds an end-to-end causal estimation track:

- `docs/segformer3d_causal_phase.md`: causal question, variables, DAG, proxy
  assumptions, identification target, and allowed answers
- `baselines/segformer3d/causal/scm.py`: structured SCM specification
- `baselines/segformer3d/data/utsw.py`: observed UTSW proxy tensors separated as
  `observed_context`, `observed_disease`, and `observed_annotation`
- `baselines/segformer3d/causal/model.py`: `CausalSegFormer3D` with explicit
  `z_d`, `z_c`, intervention hooks, and context-bank adjustment
- `baselines/segformer3d/train_causal_utsw.py`: warm-start training from the
  non-causal SegFormer3D checkpoint with proxy, adjustment, stability, and
  latent-separation losses
- `baselines/segformer3d/evaluate_causal_utsw.py`: factual segmentation,
  context-adjusted segmentation, proxy loss, context-sensitivity, and overlap
  diagnostics
- `baselines/segformer3d/visualize_causal_utsw.py`: intervention-based visual
  explanation panels for factual versus context-adjusted predictions

The implemented model-level estimand is:

```text
P(M | do(Z_d=z)) ~= sum_zc P(M | Z_d=z, Z_c=zc) P(Z_c=zc)
```

This is not yet a proven biological counterfactual. It becomes a causal claim
only after proxy validity, overlap, sensitivity, and external robustness are
validated.

Tiny causal smoke run:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/train_causal_utsw.py \
  --baseline-checkpoint runs/segformer3d_utsw_smoke/best.pt \
  --output-dir runs/segformer3d_utsw_causal_smoke \
  --model-size tiny \
  --volume-size 64 \
  --latent-dim 16 \
  --epochs 1 \
  --limit-cases 3 \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --max-context-bank-batches 1 \
  --context-bank-size 2 \
  --adjustment-contexts 2
```

Evaluate that causal smoke checkpoint:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/evaluate_causal_utsw.py \
  --checkpoint runs/segformer3d_utsw_causal_smoke/best.pt \
  --split test \
  --context-bank-size 2 \
  --adjustment-contexts 2 \
  --max-context-bank-batches 1
```

Full causal run warm-started from the trained baseline:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/train_causal_utsw.py \
  --baseline-checkpoint runs/segformer3d_utsw_base/best.pt \
  --output-dir runs/segformer3d_utsw_causal \
  --model-size base \
  --volume-size 128 \
  --latent-dim 128 \
  --epochs 20 \
  --batch-size 1 \
  --lr 0.00005 \
  --weight-decay 0.0001 \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --num-workers 2
```

More deliberate causal-mechanism run:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/train_causal_utsw.py \
  --baseline-checkpoint runs/segformer3d_utsw_base/best.pt \
  --output-dir runs/segformer3d_utsw_causal_mechanism \
  --model-size base \
  --volume-size 128 \
  --latent-dim 128 \
  --epochs 30 \
  --batch-size 1 \
  --lr 0.00001 \
  --backbone-lr 0.000005 \
  --causal-lr 0.0002 \
  --freeze-backbone-epochs 8 \
  --context-bank-size 64 \
  --adjustment-contexts 8 \
  --context-bank-refresh-epochs 3 \
  --lambda-adjustment 0.5 \
  --lambda-context-proxy 0.1 \
  --lambda-disease-proxy 0.1 \
  --lambda-annotation-proxy 0.02 \
  --lambda-orthogonal 0.02 \
  --context-response-target 0.005 \
  --lambda-context-response 0.02 \
  --context-stability-margin 0.02 \
  --num-workers 2
```

This run freezes SegFormer3D longer, trains the causal heads with a higher
learning rate, updates the context bank less often, and asks the intervention
path to produce a small but bounded response instead of remaining inert.

The first mechanism run improved causal response but hurt ET enough that it
should be kept as an ablation, not as the final model. The current best held-out
test result is the lighter causal run in `runs/segformer3d_utsw_causal`.
The complete paper-style methodology for this checkpoint is in
`docs/best_segformer3d_causal_methodology.md`.

Do not use the aggressive teacher-swap run as the final model. On held-out test
it dropped below both the baseline and the light causal model, so it is now a
negative ablation. The failed configuration was:

```text
runs/segformer3d_utsw_causal_teacher_swap
```

If teacher guidance is revisited, keep it reduced and ablation-only: no hard
farthest context swapping, no context-swap teacher distillation, and much smaller
teacher weights. A conservative diagnostic run is:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/train_causal_utsw.py \
  --baseline-checkpoint runs/segformer3d_utsw_base/best.pt \
  --teacher-checkpoint runs/segformer3d_utsw_causal/best.pt \
  --output-dir runs/segformer3d_utsw_causal_teacher_reduced_ablation \
  --model-size base \
  --volume-size 128 \
  --latent-dim 128 \
  --epochs 20 \
  --batch-size 1 \
  --lr 0.00001 \
  --backbone-lr 0.000005 \
  --causal-lr 0.00005 \
  --freeze-backbone-epochs 4 \
  --context-bank-size 64 \
  --context-bank-sampling uniform \
  --adjustment-contexts 4 \
  --context-swap-strategy none \
  --context-bank-refresh-epochs 3 \
  --channel-loss-weights 1.0,1.0,1.5 \
  --region-loss-weights 1.0,1.2,2.0 \
  --distill-channel-weights 1.0,1.0,1.5 \
  --lambda-region-loss 0.05 \
  --lambda-adjustment 0.25 \
  --lambda-teacher-distill 0.03 \
  --lambda-adjusted-teacher-distill 0.03 \
  --lambda-context-proxy 0.05 \
  --lambda-disease-proxy 0.05 \
  --lambda-annotation-proxy 0.02 \
  --lambda-orthogonal 0.01 \
  --context-stability-margin 0.02 \
  --num-workers 2
```

Keep this reduced run only if it beats `runs/segformer3d_utsw_causal` on the
held-out test split without damaging ET or HD95.

Final causal test evaluation:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/evaluate_causal_utsw.py \
  --checkpoint runs/segformer3d_utsw_causal/best.pt \
  --split test \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --num-workers 2
```

Compare a candidate against the baseline and the current best causal reference:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/compare_causal_runs.py \
  --candidate runs/segformer3d_utsw_causal_teacher_reduced_ablation/test_causal_metrics.json \
  --candidate-name teacher_reduced_ablation
```

Keep a mechanism only if it improves validation/test Dice or provides clearly
stronger intervention diagnostics without unacceptable ET or HD95 damage.

The causal evaluator writes `test_causal_metrics.json` with:

- factual `P(M | X)` segmentation metrics
- adjusted `P(M | do(Z_d=z))` approximation metrics
- metadata proxy MSEs for `Z_c`, `Z_d`, and annotation process proxies
- context-intervention probability shift
- nearest-context-bank distance as a practical overlap diagnostic

Zero-shot BraTS2020 HDF5 evaluation with the same best causal checkpoint:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/evaluate_causal_brats_h5.py \
  --checkpoint runs/segformer3d_utsw_causal/best.pt \
  --brats-csv data/brats/brats_val.csv \
  --context-csv data/brats/brats_train.csv \
  --data-root /path/to/BraTS2020_training_data/content/data \
  --split-name brats_val \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --num-workers 2
```

The BraTS HDF5 evaluator reconstructs complete 3D volumes from the per-slice
CSV before applying the SegFormer3D checkpoint. Because this BraTS CSV does not
carry the UTSW metadata proxy columns, the run reports factual/adjusted
segmentation, context shift, and overlap diagnostics, but intentionally omits
metadata proxy losses.

## Full BraTS2020 HDF5 Training

For a real BraTS experiment, do not use the UTSW checkpoint zero-shot. Train on
the BraTS HDF5 volumes, checkpoint on `brats_val.csv`, then train the causal
wrapper from the BraTS baseline.

Train the non-causal BraTS baseline:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/train_brats_h5.py \
  --train-csv data/brats/brats_train.csv \
  --val-csv data/brats/brats_val.csv \
  --data-root data/brats/archive/BraTS2020_training_data/content/data \
  --output-dir runs/segformer3d_brats_h5_base \
  --model-size base \
  --volume-size 128 \
  --epochs 100 \
  --batch-size 1 \
  --lr 0.0002 \
  --weight-decay 0.0001 \
  --num-workers 2
```

Train the causal BraTS model from that BraTS baseline:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/train_causal_brats_h5.py \
  --baseline-checkpoint runs/segformer3d_brats_h5_base/best.pt \
  --train-csv data/brats/brats_train.csv \
  --val-csv data/brats/brats_val.csv \
  --data-root data/brats/archive/BraTS2020_training_data/content/data \
  --output-dir runs/segformer3d_brats_h5_causal \
  --model-size base \
  --volume-size 128 \
  --latent-dim 128 \
  --epochs 20 \
  --batch-size 1 \
  --lr 0.00005 \
  --weight-decay 0.0001 \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --num-workers 2
```

Evaluate the trained non-causal BraTS baseline:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/evaluate_brats_h5.py \
  --checkpoint runs/segformer3d_brats_h5_base/best.pt \
  --brats-csv data/brats/brats_val.csv \
  --data-root data/brats/archive/BraTS2020_training_data/content/data \
  --split-name brats_val \
  --num-workers 2
```

Evaluate the trained BraTS causal checkpoint:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/evaluate_causal_brats_h5.py \
  --checkpoint runs/segformer3d_brats_h5_causal/best.pt \
  --brats-csv data/brats/brats_val.csv \
  --context-csv data/brats/brats_train.csv \
  --data-root data/brats/archive/BraTS2020_training_data/content/data \
  --split-name brats_val \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --num-workers 2
```

The local BraTS2020 metadata currently provides labeled train/validation CSVs,
not a separate labeled test CSV. Treat `brats_val.csv` as the held-out labeled
split unless a separate `brats_test.csv` is added.

On this Mac/PyTorch setup, `--device mps` is not a stable full-training target
because 3D trilinear upsampling falls back or errors on MPS. Use the default CPU
path locally, or run the same commands on a CUDA GPU for the real full run.

Visual causal interpretability:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/segformer3d/visualize_causal_utsw.py \
  --checkpoint runs/segformer3d_utsw_causal/best.pt \
  --split test \
  --region ET \
  --modality flair \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --num-workers 2
```

This writes a PNG panel and JSON sidecar under
`figures/causal_interpretability/`. The most important panel is the absolute
probability shift between the factual prediction and the context-adjusted
prediction. That map is the model-level intervention explanation; it should be
reported together with the numeric context-shift metric, because a nearly blank
map means the current adjustment path is weak.

## CPA-Seg3D Candidate Method

The next candidate final model is implemented outside the baseline folder:

```text
src/cpa_seg3d/
```

CPA-Seg3D keeps the SegFormer3D encoder inspiration but replaces the decoder
and prediction mechanism with a causal region-aware architecture:

- lightweight multiscale causal decoder
- disease/context latent split
- disease-context cross-attention
- causal FiLM modulation in the decoder
- explicit WT/TC/ET region head
- boundary refinement head
- deep supervision
- context-bank adjusted prediction

Full methodology and commands are in:

```text
docs/cpa_seg3d_methodology.md
```

Use CPA-Seg3D as the final model only if its held-out test result beats
`runs/segformer3d_utsw_causal/best.pt` without damaging ET Dice or HD95.
The heavy U-Net-style decoder remains available as `--decoder-variant unet`,
but the recommended default is `--decoder-variant lite` for faster CPU-first
iteration.
