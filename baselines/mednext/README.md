# MedNeXt Baseline

This folder contains a clean MedNeXt-style 3D segmentation baseline for direct
comparison with the SegFormer3D baseline on the same UTSW and BraTS2020 splits.

The model is a full 3D MedNeXt encoder-decoder using the local MedNeXt blocks in
`src/crn/mednext_blocks.py`, with S/B/M/L presets, kernel size 3 or 5, optional
deep supervision, and the same multilabel BraTS loss/metrics used by the
SegFormer3D baseline.

## UTSW

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_utsw.py \
  --data-root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
  --output-dir runs/mednext_utsw_s_k3 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 64 \
  --epochs 100 \
  --batch-size 1 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --num-workers 2
```

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/evaluate_utsw.py \
  --checkpoint runs/mednext_utsw_s_k3/best.pt \
  --split test \
  --output-json runs/mednext_utsw_s_k3/test_metrics.json \
  --num-workers 2
```

Train causal MedNeXt from the trained MedNeXt baseline:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_causal_utsw.py \
  --baseline-checkpoint runs/mednext_utsw_s_k3/best.pt \
  --data-root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
  --output-dir runs/mednext_utsw_causal_s_k3 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 64 \
  --latent-dim 128 \
  --epochs 20 \
  --batch-size 1 \
  --lr 0.00005 \
  --weight-decay 0.0001 \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --num-workers 2
```

Evaluate causal MedNeXt on the held-out UTSW test split:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/evaluate_causal_utsw.py \
  --checkpoint runs/mednext_utsw_causal_s_k3/best.pt \
  --split test \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --output-json runs/mednext_utsw_causal_s_k3/test_causal_metrics.json \
  --num-workers 2
```

### Revised Proxy-Adversarial Causal Run

The current causal adjustment was useful but produced a very small explicit
intervention shift. The revised model keeps the MedNeXt-S backbone fixed as
the segmentation engine and makes the causal path learn through a 3D adaptation
of SDD/CITE: `z_t`/`z_c`/`z_d` treatment-context-outcome branches, SDD joint
and branch treatment/outcome heads, SDD KL disentanglement, bank-assisted
imbalance, and CITE propensity-bank positives/negatives. See
`docs/revised_causal_mednext_methodology.md`.

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_causal_utsw.py \
  --baseline-checkpoint runs/mednext_utsw_s_k3/best.pt \
  --data-root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
  --output-dir runs/mednext_utsw_causal_sdd_cite_v2 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 64 \
  --latent-dim 128 \
  --epochs 12 \
  --batch-size 1 \
  --lr 0.00005 \
  --backbone-lr 0.00001 \
  --causal-lr 0.0002 \
  --freeze-backbone-epochs 4 \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --max-context-bank-batches 64 \
  --context-bank-refresh-epochs 2 \
  --lambda-adjustment 0.30 \
  --lambda-region-loss 0.10 \
  --lambda-region-volume-proxy 0.05 \
  --lambda-context-from-disease-adversary 0.02 \
  --lambda-disease-from-context-adversary 0.02 \
  --lambda-region-from-context-adversary 0.02 \
  --region-volume-scale 1000 \
  --lambda-sdd-context-teacher 0.03 \
  --lambda-sdd-region-teacher 0.05 \
  --lambda-sdd-context-distill 0.02 \
  --lambda-sdd-region-distill 0.03 \
  --lambda-sdd-treatment 0.05 \
  --lambda-sdd-treatment-disentangle 0.02 \
  --lambda-sdd-outcome 0.05 \
  --lambda-sdd-outcome-disentangle 0.03 \
  --lambda-sdd-imbalance 0.01 \
  --lambda-cite-contrastive 0.05 \
  --cite-temperature 0.2 \
  --cite-bank-negatives 32 \
  --contrastive-dim 64 \
  --lambda-context-response 0.005 \
  --context-response-target 0.002 \
  --adversary-strength 1.0 \
  --causal-residual-scale 0.2 \
  --num-workers 2
```

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/evaluate_causal_utsw.py \
  --checkpoint runs/mednext_utsw_causal_sdd_cite_v2/best.pt \
  --split test \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --output-json runs/mednext_utsw_causal_sdd_cite_v2/test_causal_metrics.json \
  --num-workers 2
```

### Causal Style-Intervention Run

If SDD/CITE improves only marginally, use this stronger causal run. It treats
scanner/acquisition appearance as the style variable `S` and trains the model
against synthetic `do(S=s')` interventions that preserve the tumor mask:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_causal_utsw.py \
  --baseline-checkpoint runs/mednext_utsw_s_k3/best.pt \
  --data-root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
  --output-dir runs/mednext_utsw_causal_style_v1 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 64 \
  --latent-dim 128 \
  --epochs 12 \
  --batch-size 1 \
  --lr 0.00005 \
  --backbone-lr 0.00001 \
  --causal-lr 0.0002 \
  --freeze-backbone-epochs 2 \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --max-context-bank-batches 64 \
  --context-bank-refresh-epochs 2 \
  --lambda-adjustment 0.20 \
  --lambda-region-loss 0.10 \
  --lambda-region-volume-proxy 0.05 \
  --lambda-sdd-treatment 0.03 \
  --lambda-sdd-treatment-disentangle 0.01 \
  --lambda-sdd-outcome 0.03 \
  --lambda-sdd-outcome-disentangle 0.02 \
  --lambda-sdd-imbalance 0.005 \
  --lambda-cite-contrastive 0.03 \
  --style-intervention-prob 0.75 \
  --lambda-style-intervention-seg 0.35 \
  --lambda-style-intervention-consistency 0.12 \
  --lambda-style-disease-invariance 0.03 \
  --lambda-style-context-response 0.01 \
  --style-context-response-target 1.0 \
  --style-scale-range 0.75,1.30 \
  --style-shift-range=-0.20,0.20 \
  --style-gamma-range 0.70,1.45 \
  --style-bias-strength 0.20 \
  --style-noise-std 0.03 \
  --cite-temperature 0.2 \
  --cite-bank-negatives 32 \
  --num-workers 2
```

Evaluate both ordinary factual output and causal style-intervention TTA:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/evaluate_causal_utsw.py \
  --checkpoint runs/mednext_utsw_causal_style_v1/best.pt \
  --split test \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --style-tta-samples 4 \
  --style-scale-range 0.85,1.15 \
  --style-shift-range=-0.10,0.10 \
  --style-gamma-range 0.85,1.20 \
  --style-bias-strength 0.10 \
  --style-noise-std 0.01 \
  --output-json runs/mednext_utsw_causal_style_v1/test_causal_style_tta_metrics.json \
  --num-workers 2
```

### Causal Style + Feature Intervention V2

This is the stronger mechanism-level follow-up when `style_v1` plateaus. It
adds two causal interventions:

- RandConv appearance intervention, following causality-inspired single-source
  domain generalization: the acquisition style is changed by random shallow
  3D convolutions while anatomy and mask are preserved.
- Causal feature masking, following ACN-style causal random masking: decoder
  features are randomly masked and must still predict the same tumor, reducing
  reliance on fragile non-causal activations.

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_causal_utsw.py \
  --baseline-checkpoint runs/mednext_utsw_s_k3/best.pt \
  --data-root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
  --output-dir runs/mednext_utsw_causal_style_feature_v2 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 64 \
  --latent-dim 128 \
  --epochs 12 \
  --batch-size 1 \
  --lr 0.00005 \
  --backbone-lr 0.00001 \
  --causal-lr 0.0002 \
  --freeze-backbone-epochs 2 \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --max-context-bank-batches 64 \
  --context-bank-refresh-epochs 2 \
  --lambda-adjustment 0.20 \
  --lambda-region-loss 0.10 \
  --lambda-region-volume-proxy 0.05 \
  --lambda-sdd-treatment 0.03 \
  --lambda-sdd-treatment-disentangle 0.01 \
  --lambda-sdd-outcome 0.03 \
  --lambda-sdd-outcome-disentangle 0.02 \
  --lambda-sdd-imbalance 0.005 \
  --lambda-cite-contrastive 0.03 \
  --style-intervention-prob 0.75 \
  --lambda-style-intervention-seg 0.25 \
  --lambda-style-intervention-consistency 0.10 \
  --lambda-style-disease-invariance 0.03 \
  --lambda-style-context-response 0.01 \
  --style-context-response-target 1.0 \
  --style-scale-range 0.75,1.30 \
  --style-shift-range=-0.20,0.20 \
  --style-gamma-range 0.70,1.45 \
  --style-bias-strength 0.20 \
  --style-noise-std 0.03 \
  --style-randconv-layers 1 \
  --style-randconv-strength 0.45 \
  --feature-intervention-prob 0.75 \
  --feature-mask-prob 0.15 \
  --feature-mask-block-size 4 \
  --lambda-feature-intervention-seg 0.20 \
  --lambda-feature-intervention-consistency 0.08 \
  --cite-temperature 0.2 \
  --cite-bank-negatives 32 \
  --num-workers 2
```

Stop this run early if validation factual Dice is still below `0.84` after
epoch 5. Continue only if it is clearly moving toward the required `>=0.87`
test Dice target.

### Spatial Causal Mediator V3

If v2 still sits near `0.84`, the remaining caveat is that `z_d` and `z_c`
are global bottleneck vectors. They regularize the scan but do not directly
localize tumor voxels. V3 adds a spatial causal mediator:

- a disease-attention map supervised by WT,
- an explicit WT/TC/ET spatial region head,
- a lesion-gated causal refiner that directly adjusts subregion logits,
- optional sparsity outside the disease-attention map.

This is the first mechanism in this line that directly changes the voxel mask
through a causal disease mediator, rather than only changing global latents.

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_causal_utsw.py \
  --baseline-checkpoint runs/mednext_utsw_s_k3/best.pt \
  --data-root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
  --output-dir runs/mednext_utsw_causal_spatial_mediator_v3 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 64 \
  --latent-dim 128 \
  --epochs 12 \
  --batch-size 1 \
  --lr 0.00005 \
  --backbone-lr 0.00001 \
  --causal-lr 0.0002 \
  --freeze-backbone-epochs 2 \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --max-context-bank-batches 64 \
  --context-bank-refresh-epochs 2 \
  --lambda-adjustment 0.20 \
  --lambda-region-loss 0.10 \
  --lambda-region-volume-proxy 0.05 \
  --lambda-spatial-disease-attention 0.30 \
  --lambda-spatial-region-head 0.25 \
  --lambda-causal-refiner-sparsity 0.005 \
  --spatial-refiner-scale 0.75 \
  --lambda-sdd-treatment 0.03 \
  --lambda-sdd-treatment-disentangle 0.01 \
  --lambda-sdd-outcome 0.03 \
  --lambda-sdd-outcome-disentangle 0.02 \
  --lambda-sdd-imbalance 0.005 \
  --lambda-cite-contrastive 0.03 \
  --style-intervention-prob 0.75 \
  --style-randconv-layers 1 \
  --style-randconv-strength 0.45 \
  --feature-intervention-prob 0.75 \
  --feature-mask-prob 0.15 \
  --lambda-style-intervention-seg 0.20 \
  --lambda-style-intervention-consistency 0.08 \
  --lambda-style-disease-invariance 0.03 \
  --lambda-style-context-response 0.01 \
  --lambda-feature-intervention-seg 0.15 \
  --lambda-feature-intervention-consistency 0.06 \
  --style-shift-range=-0.20,0.20 \
  --num-workers 2
```

Stop this run if the validation factual Dice is still below `0.845` by epoch
5. Keep it only if it breaks the plateau and moves toward the `0.87` target.

### Hierarchical Region-Mediator Fusion V4

V3 still leaves one structural gap: WT/TC/ET are supervised, but final
prediction is NCR/edema/ET. V4 converts the causal region mediator into
subregion priors:

```text
NCR/NET ~= TC * (1 - ET)
edema   ~= WT * (1 - TC)
ET      ~= ET
```

Those priors are fused into the subregion logits under the disease-attention
gate and also supervised directly. This makes the causal region head an
operational part of the prediction, not only an auxiliary explanation.

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_causal_utsw.py \
  --baseline-checkpoint runs/mednext_utsw_s_k3/best.pt \
  --data-root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
  --output-dir runs/mednext_utsw_causal_hierarchical_v4 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 64 \
  --latent-dim 128 \
  --epochs 12 \
  --batch-size 1 \
  --lr 0.00005 \
  --backbone-lr 0.00001 \
  --causal-lr 0.0002 \
  --freeze-backbone-epochs 2 \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --max-context-bank-batches 64 \
  --context-bank-refresh-epochs 2 \
  --lambda-adjustment 0.20 \
  --lambda-region-loss 0.10 \
  --lambda-region-volume-proxy 0.05 \
  --lambda-spatial-disease-attention 0.30 \
  --lambda-spatial-region-head 0.25 \
  --lambda-subregion-prior 0.20 \
  --lambda-causal-refiner-sparsity 0.005 \
  --spatial-refiner-scale 0.75 \
  --region-fusion-scale 0.25 \
  --lambda-sdd-treatment 0.03 \
  --lambda-sdd-treatment-disentangle 0.01 \
  --lambda-sdd-outcome 0.03 \
  --lambda-sdd-outcome-disentangle 0.02 \
  --lambda-sdd-imbalance 0.005 \
  --lambda-cite-contrastive 0.03 \
  --style-intervention-prob 0.75 \
  --style-randconv-layers 1 \
  --style-randconv-strength 0.45 \
  --feature-intervention-prob 0.75 \
  --feature-mask-prob 0.15 \
  --lambda-style-intervention-seg 0.20 \
  --lambda-style-intervention-consistency 0.08 \
  --lambda-style-disease-invariance 0.03 \
  --lambda-style-context-response 0.01 \
  --lambda-feature-intervention-seg 0.15 \
  --lambda-feature-intervention-consistency 0.06 \
  --style-shift-range=-0.20,0.20 \
  --num-workers 2
```

Stop V4 if validation factual Dice is still below `0.845` by epoch 5. If it
does not clearly beat V3, remove the hierarchy fusion and move to a different
causal mechanism.

### Semantic-Prototype Causal Mediator V5

V5 addresses the main caveat left by V4: a global causal latent and a WT/TC/ET
auxiliary head can still have too little voxel-level control over the final
mask. The model now learns dense semantic prototypes for background,
NCR/NET, edema, and ET. Decoder voxels are pulled toward those prototypes,
prototype-derived subregion logits are fused into the prediction under the
disease-attention gate, and a boundary mediator is exposed to the causal
refiner.

This is the first mechanism to try if the target is a visible improvement over
the MedNeXt baseline, because it gives the causal component a direct dense
route into the segmentation output.

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_causal_utsw.py \
  --baseline-checkpoint runs/mednext_utsw_s_k3/best.pt \
  --data-root "data/brats/PKG - UTSW-Glioma/UTSW-Glioma" \
  --output-dir runs/mednext_utsw_causal_prototype_v5 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 64 \
  --latent-dim 128 \
  --epochs 12 \
  --batch-size 1 \
  --lr 0.00005 \
  --backbone-lr 0.00001 \
  --causal-lr 0.0002 \
  --freeze-backbone-epochs 2 \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --max-context-bank-batches 64 \
  --context-bank-refresh-epochs 2 \
  --lambda-adjustment 0.20 \
  --lambda-region-loss 0.10 \
  --lambda-region-volume-proxy 0.05 \
  --lambda-spatial-disease-attention 0.30 \
  --lambda-spatial-region-head 0.25 \
  --lambda-subregion-prior 0.15 \
  --lambda-prototype-mediator 0.25 \
  --lambda-boundary-mediator 0.10 \
  --lambda-causal-refiner-sparsity 0.005 \
  --spatial-refiner-scale 0.75 \
  --region-fusion-scale 0.20 \
  --prototype-fusion-scale 0.35 \
  --lambda-sdd-treatment 0.03 \
  --lambda-sdd-treatment-disentangle 0.01 \
  --lambda-sdd-outcome 0.03 \
  --lambda-sdd-outcome-disentangle 0.02 \
  --lambda-sdd-imbalance 0.005 \
  --lambda-cite-contrastive 0.03 \
  --style-intervention-prob 0.75 \
  --style-randconv-layers 1 \
  --style-randconv-strength 0.45 \
  --feature-intervention-prob 0.75 \
  --feature-mask-prob 0.15 \
  --lambda-style-intervention-seg 0.20 \
  --lambda-style-intervention-consistency 0.08 \
  --lambda-style-disease-invariance 0.03 \
  --lambda-style-context-response 0.01 \
  --lambda-feature-intervention-seg 0.15 \
  --lambda-feature-intervention-consistency 0.06 \
  --style-shift-range=-0.20,0.20 \
  --num-workers 2
```

Evaluate with:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/evaluate_causal_utsw.py \
  --checkpoint runs/mednext_utsw_causal_prototype_v5/best.pt \
  --split test \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 2 \
  --output-json runs/mednext_utsw_causal_prototype_v5/test_causal_metrics.json \
  --num-workers 2
```

Keep V5 only if it beats the baseline test Dice by a clear margin and improves
at least one difficult region, especially TC or ET. If validation is still
below `0.845` by epoch 5, or if prototype fusion improves WT while hurting
TC/ET, discard this branch and move to lesion copy-paste counterfactuals.

## BraTS2020 HDF5

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_brats_h5.py \
  --train-csv data/brats/brats_train.csv \
  --val-csv data/brats/brats_val.csv \
  --data-root data/brats/archive/BraTS2020_training_data/content/data \
  --output-dir runs/mednext_brats_h5_s_k3 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 128 \
  --epochs 100 \
  --batch-size 1 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --num-workers 2
```

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/evaluate_brats_h5.py \
  --checkpoint runs/mednext_brats_h5_s_k3/best.pt \
  --brats-csv data/brats/brats_val.csv \
  --data-root data/brats/archive/BraTS2020_training_data/content/data \
  --split-name brats_val \
  --output-json runs/mednext_brats_h5_s_k3/brats_val_metrics.json \
  --num-workers 2
```

Train causal MedNeXt from the trained BraTS MedNeXt baseline:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/train_causal_brats_h5.py \
  --baseline-checkpoint runs/mednext_brats_h5_s_k3/best.pt \
  --train-csv data/brats/brats_train.csv \
  --val-csv data/brats/brats_val.csv \
  --data-root data/brats/archive/BraTS2020_training_data/content/data \
  --output-dir runs/mednext_brats_h5_causal_s_k3 \
  --model-id S \
  --kernel-size 3 \
  --volume-size 128 \
  --latent-dim 128 \
  --epochs 20 \
  --batch-size 1 \
  --lr 0.00005 \
  --weight-decay 0.0001 \
  --context-bank-size 64 \
  --context-bank-sampling farthest \
  --adjustment-contexts 4 \
  --lambda-adjustment 0.25 \
  --lambda-region-loss 0.10 \
  --lambda-region-volume-proxy 0.05 \
  --lambda-sdd-outcome 0.05 \
  --lambda-sdd-outcome-disentangle 0.03 \
  --lambda-cite-contrastive 0.05 \
  --cite-temperature 0.2 \
  --cite-bank-negatives 32 \
  --num-workers 2
```

Evaluate causal MedNeXt on the BraTS validation split:

```bash
PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  baselines/mednext/evaluate_causal_brats_h5.py \
  --checkpoint runs/mednext_brats_h5_causal_s_k3/best.pt \
  --brats-csv data/brats/brats_val.csv \
  --context-csv data/brats/brats_train.csv \
  --data-root data/brats/archive/BraTS2020_training_data/content/data \
  --split-name brats_val \
  --context-bank-size 64 \
  --adjustment-contexts 4 \
  --output-json runs/mednext_brats_h5_causal_s_k3/brats_val_causal_metrics.json \
  --num-workers 2
```

Start with `model-id S`, `kernel-size 3` for speed. If it beats or matches the
SegFormer3D baseline, run `model-id B`. Kernel 5 is more expensive and should be
tested after kernel 3 is stable.
