# Experiment Plan

## Research Questions

1. Does latent backdoor adjustment improve disease prediction under context shift compared with a plain supervised model?
2. Does bounded counterfactual stability avoid the causal inconsistency of strict confounder removal?
3. Does the learned factorization produce interpretable counterfactuals without sacrificing task performance?

## Primary Datasets

CheXpert classification:

- Multi-label chest X-ray prediction.
- Metrics: per-label AUROC, macro AUROC, AUPRC, calibration error.
- If metadata is available: subgroup AUROC and equalized-odds gaps by sex, age bin, scanner/view, or site.

BraTS segmentation:

- Tumor segmentation with optional tumor-present classification.
- Metrics: Dice, Dice+BCE validation loss, Hausdorff distance if added later.
- Counterfactual checks: prediction stability under context swaps and disease-code swaps.

## Baselines

- ERM CNN: same encoder capacity, no latent factorization.
- CRN old objective: heads consume `z_d` only with strict context invariance.
- CRN no adjustment: heads consume `[z_d, z_c]`, no backdoor averaging.
- CRN no bounded stability: adjustment without context-swap stability.
- CRN no reconstruction: tests whether reconstruction improves factor preservation.

## Ablations

- Context stability margin: `0.00`, `0.05`, `0.10`, `0.20`.
- Number of context samples for adjustment: `batch`, `2`, `4`, `8`.
- Latent dimension: `64`, `128`, `256`.
- Decorrelation weight: `0`, `0.001`, `0.01`, `0.1`.

## Expected Claims

- The revised CRN is causally consistent because it estimates `p(y | do(z_d))` via latent adjustment rather than deleting `z_c`.
- Context swapping is an explanation and regularization device, not proof that context has no effect.
- The method exposes a controllable tradeoff between context sensitivity and disease-factor stability.

