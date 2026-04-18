# Experiment Plan

## Current Scope

The project is now primarily a **BraTS segmentation** study rather than a mixed segmentation/classification project. The main scientific question is whether causal effect estimation under latent confounding can improve brain tumor segmentation while producing intervention-based explanations.

## Current Best Model

Current strongest run:

- config: `configs/crn_brats_segonly_unet_causal_contrastive_continue.yaml`
- best epoch: `6`
- best threshold: `0.4`

Current best validation volume metrics:

- mean Dice: `0.7797`
- mean HD95: `11.2457`
- WT Dice: `0.8640`
- TC Dice: `0.7678`
- ET Dice: `0.7074`

Compared with the earlier causal-region baseline:

- mean Dice: `0.7613 -> 0.7797`
- mean HD95: `13.2669 -> 11.2457`

## Current Research Questions

1. Does **region-level latent backdoor adjustment** improve BraTS volume segmentation under latent confounding?
2. Does **matched disease counterfactual supervision** improve tumor-region quality beyond adjustment alone?
3. Does **lesion-aware counterfactual contrastive learning** provide additional value after adjustment and disease-swap supervision?
4. Can the model provide **causal output-image explanations** through disease and context interventions?

## Current Baselines

- Causal-region baseline:
  - region adjustment
  - region context stability
  - region disease swap
  - no lesion-aware contrastive term

- Full contrastive model:
  - region adjustment
  - region context stability
  - region disease swap
  - lesion-aware counterfactual contrastive term

## Current Ablation Findings

From the continuation ablations:

| Ablation | Mean Dice | Mean HD95 | Takeaway |
|---|---:|---:|---|
| Full model | `0.7797` | `11.2457` | strongest overall |
| `- region_adjustment` | `0.7743` | `12.7394` | largest causal drop |
| `- region_disease_swap` | `0.7767` | `12.7706` | second most important drop |
| `- contrastive` | `0.7791` | `11.6777` | modest Dice drop, clearer HD95 drop |
| `- region_context_stability` | `0.7799` | `11.6662` | Dice neutral/slightly up, HD95 worse |

Current interpretation:

- `region_adjustment` is the most important causal mechanism
- `region_disease_swap` is the second most important causal mechanism
- contrastive refinement helps, especially on volume coherence / HD95
- context stability is useful mainly as a regularizer for geometric stability

## What To Do Next

### 1. Freeze the current full model as the benchmark result

Primary checkpoint:

- `runs/brats_segonly_unet_causal_contrastive_continue/best.pt`

This should be treated as the current paper result unless a stronger run is produced.

### 2. Run one focused tuning experiment on context stability

Motivation:

- removing `region_cf_stability` slightly helps Dice
- but hurts HD95

Recommended next test:

- reduce `lambda_region_cf_stability` from `0.04` to `0.02`

This is a more justified next move than adding a new mechanism.

### 3. Strengthen the evidence for contrastive learning

The continuation ablations start from an already contrastive checkpoint, so they mainly test late-stage contribution. If the paper needs a stronger causal claim for the contrastive term, run:

- with contrastive
- without contrastive

from an earlier non-contrastive checkpoint, using the same schedule.

### 4. Improve the backbone, not the causal framing

The main performance gap to `nnU-Net` is likely architectural:

- our model is still 2D
- `nnU-Net` is a heavily optimized 3D segmentation system

So if we want a major next performance jump, the right direction is:

- keep the causal mechanism
- move it onto a stronger 2.5D or 3D segmentation backbone

Implementation status:

- a new experimental 2.5D causal backbone path is now available in
  `configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml`
- it uses slice stacks in `depth` layout and a volumetric U-Net-style encoder/decoder
- it keeps the same region-adjustment, region-disease-swap, and lesion-aware contrastive objectives
- it warm-starts from the current best 2D contrastive checkpoint via kernel inflation

## Expected Claims

The current evidence supports the following paper claims:

1. The framework is causally better grounded than confounder-removal approaches because it estimates effects by **adjusting for context** rather than deleting context.
2. Region-level backdoor adjustment and disease-counterfactual supervision are the strongest causal contributors to performance.
3. Lesion-aware counterfactual contrastive learning improves the strongest model further, particularly in volume-level coherence and HD95.
4. The system provides intervention-based interpretability through output-image counterfactual panels and effect maps.
