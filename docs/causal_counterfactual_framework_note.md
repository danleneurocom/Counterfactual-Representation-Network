# Causal Counterfactual Representation Network for Brain Tumor Segmentation Under Latent Confounding

## Abstract

We present a causal counterfactual representation framework for brain tumor segmentation under latent confounding. The central motivation is that confounders in medical imaging are often not nuisance variables that should be removed from inference; rather, they can also be causal parents of the target. Consequently, a causally consistent segmentation model should condition on both disease-related and context-related factors, while estimating the intervention effect of disease factors through adjustment. Our framework learns a disease latent `z_d` and a context latent `z_c`, predicts segmentation from both latents, and approximates the interventional estimand `p(m | do(z_d))` through latent backdoor adjustment. We further introduce matched counterfactual context swaps, matched disease swaps, and a lesion-aware counterfactual contrastive loss that aligns factual lesion features with backdoor-adjusted same-disease features while separating them from matched disease-counterfactual features. Finally, we provide intervention-based interpretability through causal counterfactual image panels and effect maps. The resulting framework is designed to be both causally grounded and visually explainable in the output image space.

## 1. Introduction

Brain tumor segmentation in MRI is highly sensitive to latent variation beyond pathology itself, including anatomy, scanner style, acquisition differences, and other contextual factors. Many representation-learning approaches attempt to isolate disease-related information by suppressing or removing confounders from the predictive pathway. However, this can be causally inconsistent when the confounders are not merely irrelevant noise, but also participate in the causal structure of the target.

Our work starts from a corrected causal principle: under confounding, the goal is not to remove confounders, but to adjust for them when estimating the effect of disease-related factors. This leads to a different design from confounder-removal methods. Instead of predicting from `z_d` alone, we predict from both `z_d` and `z_c`, and regularize the model so that the contribution of `z_d` is not spuriously driven by `z_c`.

The framework is developed for BraTS-style tumor segmentation, where the clinically meaningful output regions are whole tumor (WT), tumor core (TC), and enhancing tumor (ET). Because these regions are the medically relevant manifestation of disease, the causal objectives are applied not only at the latent level, but also at the BraTS region level.

## 2. Problem Setting and Causal Motivation

Let `x` denote an MRI input and `m` its tumor segmentation mask. We consider a latent causal factorization:

```text
z_d = E_d(x)
z_c = E_c(x)
```

where:

- `z_d` represents disease-related variation
- `z_c` represents contextual and confounding variation

The key modeling decision is that segmentation depends on both latent components:

```text
m_hat = f_seg(z_d, z_c)
```

This avoids the causal mistake of treating confounders as variables that must be removed from prediction. In our framing, the target quantity is the intervention effect of disease factors:

```text
p(m | do(z_d))
```

not the observational shortcut:

```text
p(m | z_d)
```

Therefore, the model should retain `z_c` in the predictive mechanism while adjusting over it when estimating the effect of `z_d`.

## 3. Framework Overview

For an input image `x`, the framework learns two encoders and three downstream components:

```text
z_d = E_d(x)
z_c = E_c(x)
m_hat = f_seg(z_d, z_c)
y_hat = f_cls(z_d, z_c)          optional classification head
x_hat = G(z_d, z_c)              reconstruction / counterfactual decoder
```

In the current BraTS configuration, the main predictive pathway is a context-conditioned U-Net segmentation decoder. The disease encoder also supplies multi-scale disease features to the decoder, while the context latent conditions the output so that predictions remain confounder-aware rather than confounder-blind.

The reconstruction branch is not treated as a separate generative objective for its own sake. Instead, it enables causal interpretability by allowing us to visualize factual and counterfactual outputs under controlled interventions on `z_d` and `z_c`.

## 4. Causal Mechanisms

### 4.1 Latent Backdoor Adjustment

The central causal estimator in the framework is latent backdoor adjustment. Rather than predicting from a single observed context, we estimate the effect of disease factors by averaging predictions over a bank of context latents:

```text
p(m | do(z_d_i)) ~= (1 / K) sum_k p(m | z_d_i, z_c^(k))
```

This preserves the role of context in prediction while approximating the intervention effect of disease. In practice, the current mainline implementation uses a context bank built from observed latent contexts and applies the adjustment directly to segmentation outputs.

### 4.2 Bounded Context Counterfactuals

Context counterfactuals are formed by holding `z_d` fixed and replacing `z_c` with a matched alternative context:

```text
m_cf_ctx = f_seg(z_d, z_c')
```

Importantly, we do not enforce strict invariance. Strict invariance would implicitly assume that context has zero causal influence, which is exactly the causal error we seek to avoid. Instead, we apply a bounded stability loss:

```text
L_ctx = max(|m_hat - m_cf_ctx| - margin, 0)
```

Small context effects are allowed; unstable, excessive context dependence is penalized.

### 4.3 Matched Disease Counterfactuals

Disease counterfactuals are formed by holding context fixed and replacing the disease factor with a matched donor disease latent:

```text
m_cf_dis = f_seg(z_d', z_c)
```

This approximates a disease intervention under controlled context. The corresponding supervision encourages the model to respond to lesion-relevant disease changes rather than contextual shortcuts.

### 4.4 BraTS Region-Aware Causal Supervision

For BraTS, the clinically meaningful outputs are the derived regions:

- `WT = NCR union ED union ET`
- `TC = NCR union ET`
- `ET = ET`

We therefore apply the causal objectives at the region level, not only at the raw channel level. This yields region-adjusted supervision, region-level bounded context stability, and region-level disease-swap supervision. The advantage is that the causal constraints are imposed on the structures that matter clinically and are used in benchmark reporting.

### 4.5 Lesion-Aware Counterfactual Contrastive Learning

Our newest mechanism is a lesion-aware counterfactual contrastive loss defined directly on segmentation features. The goal is to make disease representations more identifiable in the lesion space itself.

For each lesion-positive sample, we define:

- **anchor**: factual lesion-conditioned segmentation features
- **positive**: backdoor-adjusted same-disease features
- **hard negative**: matched disease-swapped counterfactual features

Formally, the contrastive construction is:

```text
anchor   = phi(f_seg(z_d, z_c), lesion)
positive = E_zc[ phi(f_seg(z_d, z_c), lesion) ]
negative = phi(f_seg(z_d', z_c), lesion_donor)
```

where `phi` denotes lesion-aware pooling over segmentation features. The objective pulls the factual representation toward the backdoor-adjusted same-disease representation and pushes it away from the matched disease-counterfactual representation.

This is important because it strengthens the causal mechanism itself instead of only adding another output loss. The model is encouraged to organize lesion features according to disease identity under adjustment, while remaining resistant to context-driven shortcuts.

## 5. Training Objective

The full objective combines segmentation supervision with causal regularization:

```text
L_total =
    lambda_seg * L_seg
  + lambda_region_adjustment * L_region_adjustment
  + lambda_region_cf_stability * L_region_context_stability
  + lambda_region_disease_swap * L_region_disease_swap
  + lambda_region_cf_contrastive * L_region_contrastive
  + lambda_dis * L_decorrelation
  + lambda_rec * L_reconstruction
```

where:

- `L_seg` is standard segmentation supervision
- `L_region_adjustment` supervises adjusted `p(m | do(z_d))`
- `L_region_context_stability` bounds unstable context sensitivity
- `L_region_disease_swap` supervises disease interventions under fixed context
- `L_region_contrastive` aligns factual and adjusted lesion features against hard disease counterfactuals
- `L_decorrelation` discourages trivial collapse between `z_d` and `z_c`

The main point is that the framework remains confounder-aware throughout. Nowhere does the method enforce that context should vanish from prediction.

## 6. Interpretability Through Causal Output Images

Our interpretability mechanism is intervention-based rather than purely post hoc. Instead of only producing attention or saliency maps, the framework exports image-space causal explanation panels:

1. **Factual reconstruction**
2. **Context-swapped counterfactual image**
3. **Context-swapped counterfactual segmentation**
4. **Disease-swapped counterfactual image**
5. **Disease-swapped counterfactual segmentation**
6. **Disease do-effect map**
7. **Context-shift sensitivity map**

These outputs support a causal reading of the model:

- if `z_d` is fixed and `z_c` changes, lesion predictions should remain relatively stable
- if `z_c` is fixed and `z_d` changes, lesion structure should change in a disease-relevant way

This gives a stronger notion of explainability than ordinary saliency, because the explanation is tied to explicit interventions in the learned causal representation.

## 7. What Is Novel

The novelty of the framework does not lie in using a U-Net-style decoder alone. The novelty is the causal mechanism layered on top of segmentation:

1. **Causally corrected problem formulation**  
   We frame the task as causal effect estimation under latent confounding, not confounder removal.

2. **Latent backdoor adjustment for segmentation**  
   We explicitly approximate `p(m | do(z_d))` while preserving context in the predictive pathway.

3. **Matched counterfactual regularization with bounded context influence**  
   Context swaps stabilize predictions without assuming zero context effect.

4. **Disease-counterfactual supervision under fixed context**  
   Disease swaps approximate lesion interventions in a controlled latent setting.

5. **Lesion-aware counterfactual contrastive learning**  
   We align factual lesion features with adjusted same-disease features and separate them from hard disease-counterfactual lesion features.

6. **Output-image causal interpretability**  
   The framework produces counterfactual images and causal effect maps rather than relying only on post hoc attention.

Taken together, these components form a segmentation framework whose contribution is causal consistency, intervention-based explainability, and region-aware counterfactual learning.

## 8. Current Empirical Snapshot

In the current BraTS validation setting, the strongest contrastive version of the framework improves the previous causal-region baseline on volume-level segmentation:

```text
Best swept volume mean Dice: 0.7613 -> 0.7703
Best swept volume mean HD95: 13.27  -> 13.23
WT Dice: 0.8532 -> 0.8604
TC Dice: 0.7383 -> 0.7539
ET Dice: 0.6925 -> 0.6966
```

The gain is not only numerical. The causal counterfactual panels also show the desired qualitative behavior: context interventions usually induce only small output changes, while disease interventions produce larger lesion-relevant effects.

## 9. Limitations and Next Steps

The current framework is already causally stronger than the earlier confounder-removal version, but several directions remain open:

- longer contrastive training to see whether the late-stage gains continue
- stronger ET-focused disease interventions
- tighter lesion-preservation constraints in counterfactual reconstructions
- paper-grade quantitative analysis of interpretability behavior

These are natural extensions of the same causal direction rather than departures from it.

## 10. Summary

Our framework proposes a causally grounded segmentation system in which disease and context are explicitly modeled as separate latent factors, prediction depends on both, causal effect estimation is approximated through adjustment, and explanations are produced through image-space counterfactual interventions. The central conceptual move is simple but important: under confounding, the right objective is not to remove confounders from prediction, but to adjust for them while preserving genuine causal structure.
