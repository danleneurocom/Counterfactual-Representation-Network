# Experiment Plan

## Current Scope

The project is now primarily a **BraTS segmentation** study rather than a mixed segmentation/classification project. The main scientific question is whether causal effect estimation under latent confounding can improve brain tumor segmentation while producing intervention-based explanations.

## Current Best Model

Current strongest run:

- config: `configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml`
- best epoch: `6`
- best threshold: `0.4`

Current best validation volume metrics:

- mean Dice: `0.7963`
- mean HD95: `9.6807`
- WT Dice: `0.8702`
- TC Dice: `0.7675`
- ET Dice: `0.7512`

Compared with the previous best 2D causal-contrastive model:

- mean Dice: `0.7797 -> 0.7963`
- mean HD95: `11.2457 -> 9.6807`
- WT Dice: `0.8640 -> 0.8702`
- TC Dice: `0.7678 -> 0.7675`
- ET Dice: `0.7074 -> 0.7512`

## Current Research Questions

1. Does **region-level latent bank backdoor adjustment**, reframed as proximal causal inference under assumptions (A1)–(A5) in `causal_counterfactual_framework_note.md` §4.1, improve BraTS volume segmentation under latent confounding?
2. Does **matched disease counterfactual supervision** improve tumor-region quality beyond adjustment alone?
3. Does **lesion-aware counterfactual contrastive learning** provide additional value after adjustment and disease-swap supervision?
4. Can the model provide **causal output-image explanations** through disease and context interventions?
5. **(New, ICDM-critical)** Does the learned factorization actually separate disease from context? Measure via quantitative counterfactual metrics M1 (Context Invariance), M2 (Disease-Swap Effect), M3 (Adebayo sanity check), M5 (axiomatic soundness: Composition / Reversibility / Effectiveness after Monteiro et al. 2023).
6. **(New, ICDM-critical)** Does the causal machinery earn its complexity under distribution shift? Test on three OOD protocols:
   - **BraTS-Africa** (Adewole et al. 2023) — zero-shot cross-population transfer.
   - **ROOD-MRI + TorchIO** (Boone et al. 2023; Pérez-García et al. 2021) — acquisition-corruption Dice-vs-severity curves.
   - **FeTS leave-one-institution-out** (Pati et al. 2021/2022) — worst-case and inter-site Dice variance.

## Target Numbers (for the ICDM submission)

Raise the in-distribution BraTS validation numbers into the current single-model SOTA band:

- WT Dice ≥ **0.90**
- TC Dice ≥ **0.85**
- ET Dice ≥ **0.80**

Path: swap the 2.5D CNN backbone for dual-encoder MedNeXt-L k5 (Roy et al., MICCAI 2023) while preserving the CRN causal losses and dual-encoder independence required by the proposition in §4.1. Keep the previous 2.5D configuration as an architecture-agnostic ablation.

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

- `runs/brats_segonly_unet_causal_contrastive_25d/best.pt`

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

### 4. Upgrade the backbone to dual-encoder MedNeXt-L k5

The main performance gap to single-model SOTA is architectural:

- the previous best model was 2D
- the current best is a 2.5D slice-stack U-Net
- published single-model baselines on BraTS 2021 (MedNeXt-L k5, Swin UNETR, SegResNetVAE) sit roughly 5–10 Dice points above our current numbers

The planned architectural step:

- **Headline**: dual-encoder MedNeXt-L k5 (Roy et al., MICCAI 2023; `nnunet_mednext` from MIC-DKFZ) via `backbone_mode: mednext` in the new `CounterfactualRepresentationNetwork` dispatch.
- **Ablation**: dual-encoder MedNeXt-B (`variant: B`) and dual-encoder SegResNet to support the "causal gains are architecture-agnostic" claim.
- **Preserved invariants**: `E_d` and `E_c` remain two fully independent instances; the `feature_channels` / `forward_features` contract is preserved so that `ContextualUNetDecoder` / `VolumetricContextualUNetDecoder` and the existing loss stack (`backdoor_adjusted_seg_logits`, `lesion_aware_region_descriptor`) work unchanged.
- **Warm start**: MedNeXt official BraTS weights if released; UpKern kernel transfer from k3 otherwise.

Implementation status:

- previous 2.5D causal backbone path remains the fallback:
  `configs/crn_brats_segonly_unet_causal_contrastive_25d.yaml`
- new MedNeXt configs land as:
  `configs/crn_brats_segonly_unet_causal_contrastive_mednextL.yaml`
  `configs/crn_brats_segonly_unet_causal_contrastive_mednextB.yaml`

### 5. Add quantitative counterfactual metrics

The paper currently renders counterfactuals but does not score them. Add to `src/crn/metrics.py`:

- **M1 Context Invariance.** Dice between factual and z_c-swapped segmentation. M1 → 1.0 ideal.
- **M2 Disease-Swap Effect.** Difference between Dice(disease-CF seg, target disease mask) and Dice(factual seg, target disease mask). M2 ≫ 0 ideal; must beat a random-latent-swap baseline.
- **M3 Adebayo sanity ratio** (Adebayo et al. 2018). |DSE(trained) − DSE(randomized z_d encoder)| / DSE(trained). M3 → 1.0 ideal.
- **M5 Axiomatic soundness** (Monteiro et al. 2023). Composition (null intervention), Reversibility (roundtrip cycle), Effectiveness (disease classifier agrees with target label on CF image).

### 6. Add OOD evaluation protocols

Three protocols, each with its own runner and config:

- `configs/eval_ood_brats_africa.yaml` — zero-shot inference on BraTS-Africa. Report WT/TC/ET with Dice drop Δ vs source.
- `configs/eval_ood_corruptions.yaml` — bias field, motion, ghosting, noise, SNR, contrast, resolution × 5 severities via TorchIO. Report Dice-vs-severity curves and mean Corruption Error (mCE).
- `configs/eval_ood_fets_loio.yaml` — leave-one-institution-out on FeTS 2022 partition. Report worst-case site Dice and inter-site Dice variance.

Each protocol must compare CRN against ERM (same backbone, no causal losses), nnU-Net, BigAug, DANN, and GIN+IPA.

## Expected Claims

The revised target claims are:

1. **Theory.** The framework is causally better grounded than confounder-removal approaches because it estimates effects by *adjusting* for context rather than deleting it; identifiability of the `z_d`/`z_c` factorization rests on auxiliary-variable + content/style + paired-intervention results; bank-averaging is reframed as proximal causal inference with explicit, testable assumptions (A1)–(A5).
2. **Mechanism ablations.** Region-level backdoor adjustment and disease-counterfactual supervision are the strongest causal contributors to performance; lesion-aware counterfactual contrastive learning gives an additional, particularly on HD95.
3. **In-distribution accuracy.** Dual-encoder MedNeXt-L with CRN losses reaches single-model WT ≥ 0.90 / TC ≥ 0.85 / ET ≥ 0.80 on BraTS 2021 validation, matching the single-model state of the art.
4. **Counterfactual quality.** M1, M2, M3, M5 all land inside the ranges targeted in §11 of the framework note (e.g. M1 ≥ 0.85, M3 ≥ 0.7, M5 Composition ≥ 0.95).
5. **Robustness.** The CRN exhibits systematically smaller Dice drops than ERM and BigAug baselines on BraTS-Africa, on ROOD-MRI non-causal corruptions, and on FeTS worst-case site Dice.
6. **Interpretability.** The system provides intervention-based interpretability through output-image counterfactual panels and effect maps, validated quantitatively (M5 Effectiveness).
