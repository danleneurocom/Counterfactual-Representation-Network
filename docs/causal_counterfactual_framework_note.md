# Causal Counterfactual Representation Network for Brain Tumor Segmentation Under Latent Confounding

## Abstract

We present a causal counterfactual representation framework for brain tumor segmentation under latent confounding. The central motivation is that confounders in medical imaging are often not nuisance variables that should be removed from inference; rather, they can also be causal parents of the target. Consequently, a causally consistent segmentation model should condition on both disease-related and context-related factors, while estimating the intervention effect of disease factors through adjustment. Our framework learns a disease latent `z_d` and a context latent `z_c`, predicts segmentation from both latents, and approximates the interventional estimand `p(m | do(z_d))` through a *latent bank backdoor adjustment* that we reframe as proximal causal inference in the sense of Miao–Geng–Tchetgen Tchetgen (2018) and Tchetgen Tchetgen et al. (2024), with explicit assumptions (A1)–(A5) on proxy structure, completeness, bridge existence, positivity, and bank representativeness. The `z_d`/`z_c` factorization is grounded in identifiability results from auxiliary-variable nonlinear ICA (iVAE), content/style block-identifiability, and paired-intervention causal representation learning, rather than on unsupervised disentanglement (which Locatello et al. 2019 proved is impossible). We further introduce matched counterfactual context swaps, matched disease swaps, and a lesion-aware counterfactual contrastive loss that aligns factual lesion features with backdoor-adjusted same-disease features while separating them from matched disease-counterfactual features. The framework is evaluated both in-distribution on BraTS and under cross-population (BraTS-Africa), acquisition-corruption (ROOD-MRI/TorchIO), and cross-site (FeTS) shift, with quantitative counterfactual metrics covering context invariance, disease-swap effect, an Adebayo-style sanity check, and axiomatic soundness. The resulting framework is both causally grounded and empirically accountable for the causal claims it makes.

## 1. Introduction

Brain tumor segmentation in MRI is highly sensitive to latent variation beyond pathology itself, including anatomy, scanner style, acquisition differences, and other contextual factors. Many representation-learning approaches attempt to isolate disease-related information by suppressing or removing confounders from the predictive pathway. However, this can be causally inconsistent when the confounders are not merely irrelevant noise, but also participate in the causal structure of the target.

Our work starts from a corrected causal principle: under confounding, the goal is not to remove confounders, but to adjust for them when estimating the effect of disease-related factors. This leads to a different design from confounder-removal methods. Instead of predicting from `z_d` alone, we predict from both `z_d` and `z_c`, and regularize the model so that the contribution of `z_d` is not spuriously driven by `z_c`.

The framework is developed for BraTS-style tumor segmentation, where the clinically meaningful output regions are whole tumor (WT), tumor core (TC), and enhancing tumor (ET). Because these regions are the medically relevant manifestation of disease, the causal objectives are applied not only at the latent level, but also at the BraTS region level.

## 2. Problem Setting, Causal Motivation, and Identifiability

### 2.1 Latent causal factorization

Let `x` denote an MRI input and `m` its tumor segmentation mask. We consider a latent causal factorization:

```text
z_d = E_d(x)
z_c = E_c(x)
```

where:

- `z_d` represents disease-related variation (the causal target of segmentation)
- `z_c` represents contextual variation (anatomy, scanner, acquisition, site)

The key modeling decision is that segmentation depends on both latent components:

```text
m_hat = f_seg(z_d, z_c)
```

This avoids the causal mistake of treating confounders as variables that must be removed from prediction. The target estimand is the intervention effect of disease factors:

```text
p(m | do(z_d))
```

not the observational shortcut `p(m | z_d)`. The model should retain `z_c` in the predictive mechanism while adjusting over it when estimating the effect of `z_d`.

### 2.2 Why identifiability is not free: Locatello's impossibility

A two-encoder architecture by itself does not license interpreting `z_d` and `z_c` as disease and context factors. Locatello et al. [1] prove that for any nonlinear generative model `p(x)` there exist infinitely many observationally equivalent factorizations of the latent space: an unsupervised reconstruction-based objective, no matter how carefully regularized, cannot identify which coordinates correspond to disease and which to context. Role-swapping failures (the classifier attending to `z_c`, the decoder leaking anatomy through `z_d`) are therefore not engineering accidents but the generic case without additional inductive bias or supervision. This impossibility result is the reason our framework is built around *paired and counterfactual supervision* rather than disentanglement objectives alone.

### 2.3 Identifiability via auxiliary variables and paired supervision

We ground the disease/context factorization on three identifiability results that jointly license our training signal:

- **Auxiliary-variable identifiability (iVAE, Khemakhem et al. [2]).** When latent priors are conditioned on an auxiliary variable `u` with sufficient variability, latents are identified up to a component-wise (affine) transform. In our setting, `u = disease label / region-presence` for `z_d` and `u = volume / site identifier` for `z_c`. The classification and region-presence supervision inside `L_seg` and the region-level causal terms in §5 act as the auxiliary signal that anchors each encoder to the right block of factors.
- **Content/style block-identifiability from paired data (von Kügelgen et al. [3]).** Contrastive alignment on pairs that share *content* but differ in *style* identifies the content block up to invertible transformation, even when content and style are statistically dependent. Our "same-disease different-context" counterfactual pairs, constructed by `CounterfactualMemory` and consumed by the lesion-aware contrastive loss (§4.5), play the role of augmentation pairs in this theorem. Content ↔ `z_d` and style ↔ `z_c`.
- **Weakly-supervised / interventional identifiability (Locatello et al. [4]; Brehmer–de Haan–Lippe–Cohen [5]; Ahuja et al. [6]).** Paired observations that share a random subset of factors, or paired pre/post-intervention samples, suffice for block-identifiable disentanglement. Our matched disease-swap pairs (§4.3), where only `z_d` is replaced while `z_c` is held fixed, are the imaging analogue of this paired-intervention regime.

Additionally, Kong et al. [7] establish partial identifiability under an *invariant / sparsely-changing* split: a factor that is invariant across domains (disease semantics) can be separated from a factor that varies sparsely with domain (context). This justifies the role asymmetry between `z_d` and `z_c` and underpins the out-of-distribution claims in the experimental sections.

For an ICDM audience, the best single-citation framing of this programme is Schölkopf et al. [8].

### 2.4 What we claim and what we do not

Combining these results, our identifiability claim is modest and precise: **within each block, `z_d` and `z_c` are identified up to an affine transformation**, provided the training signal realizes the auxiliary-variable / content-style / paired-intervention regimes above. We do *not* claim coordinate-level disentanglement, Shapley-style attribution of individual latent dimensions, or recovery of any ground-truth generative process. Under this weaker but defensible guarantee the downstream intervention estimand `p(m | do(z_d))` is well-posed: any affine reparameterization of `z_d` leaves it invariant.

### 2.5 Architectural preconditions for the identifiability claim

The theorems above have preconditions that must be enforced architecturally, not assumed:

- **Encoder independence.** `E_d` and `E_c` are instantiated as two separate modules with no shared parameters (`src/crn/models.py:524-537`). This prevents `z_c` from being a deterministic function of `z_d` (which would turn it into a descendant of the treatment and invalidate §4.1).
- **Block-structured supervision.** The auxiliary signals attached to each block (region-level targets on `z_d`, volume/context indices on `z_c`) must remain non-degenerate across batches, as required by the sufficient-variability conditions of [2, 3].
- **Optional independent-mechanism regularizer.** When the two encoders see the same input `x`, nothing prevents them from learning a common latent subspace. Gresele et al. [9] motivate an orthogonality penalty on the encoder Jacobians (IMA); Lachapelle et al. [10] motivate an ℓ₁ sparsity penalty on the supervision-to-latent graph. Either can be added to `L_total` as an architectural strengthening of (A1) in §4.1.

---

**Identifiability references.**
[1] F. Locatello et al. *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.* ICML 2019 (Best Paper). arXiv:1811.12359.
[2] I. Khemakhem, D. P. Kingma, R. P. Monti, A. Hyvärinen. *Variational Autoencoders and Nonlinear ICA: A Unifying Framework (iVAE).* AISTATS 2020. arXiv:1907.04809.
[3] J. von Kügelgen, Y. Sharma, L. Gresele, W. Brendel, B. Schölkopf, M. Besserve, F. Locatello. *Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style.* NeurIPS 2021. arXiv:2106.04619.
[4] F. Locatello et al. *Weakly-Supervised Disentanglement Without Compromises.* ICML 2020. arXiv:2002.02886.
[5] J. Brehmer, P. de Haan, P. Lippe, T. Cohen. *Weakly Supervised Causal Representation Learning.* NeurIPS 2022. arXiv:2203.16437.
[6] K. Ahuja, D. Mahajan, Y. Wang, Y. Bengio. *Interventional Causal Representation Learning.* ICML 2023. arXiv:2209.11924.
[7] L. Kong et al. *Partial Identifiability for Domain Adaptation.* ICML 2022. arXiv:2306.06510.
[8] B. Schölkopf et al. *Toward Causal Representation Learning.* Proc. IEEE 109(5):612–634, 2021. arXiv:2102.11107.
[9] L. Gresele, J. von Kügelgen, V. Stimper, B. Schölkopf, M. Besserve. *Independent Mechanism Analysis, a New Concept?* NeurIPS 2021. arXiv:2106.05200.
[10] S. Lachapelle et al. *Disentanglement via Mechanism Sparsity Regularization.* CLeaR 2022. arXiv:2107.10098.

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

### 4.1 Latent Bank Backdoor Adjustment as Proximal Causal Inference

#### 4.1.1 Estimator

The central causal estimator in the framework averages segmentation predictions over a bank of learned context latents:

```text
p(m | do(z_d_i)) ~= (1 / K) sum_k p(m | z_d_i, z_c^(k))
```

In implementation, the context bank is built from observed latent contexts: either within-batch (during training, via `CounterfactualMemory.context_bank`) or from the validation-set context bank constructed at evaluation time (`backdoor_adjusted_seg_logits` in `src/crn/losses.py:695-713`).

#### 4.1.2 Why this is not standard backdoor adjustment

A standard backdoor correction, as formalized by Pearl [11, Thm. 3.3.2], requires a *measured* set of variables satisfying the backdoor criterion. In our setting, `z_c` is not observed — it is *learned by an encoder* — so the correction is not a direct application of Pearl's theorem. The same structural issue is shared by the deconfounder [12] and by CEVAE [13], and has been shown to fail without additional assumptions [14, 15, 16]. We therefore reframe the estimator as **proximal causal inference** in the sense of Miao, Geng, and Tchetgen Tchetgen [17] and Tchetgen Tchetgen et al. [18]: `z_c` plays the role of a *proxy* for the true unobserved context `U`, and identification hinges on the existence of a *confounding bridge* that links the observed proxy to the effect of interest.

#### 4.1.3 Proposition (Latent Bank Backdoor Adjustment)

Let `Z_D` denote disease features, `M` the segmentation mask, and `U` the true unobserved context (anatomy, scanner, site, protocol). Assume:

- **(A1) Proxy structure.** `z_c = g(U, η)` with `η ⊥ (Z_D, M)`; equivalently, `z_c ⊥ M | (Z_D, U)`. The learned context encoder is a noisy function of the true context and nothing else.
- **(A2) Completeness.** The conditional operator `P(z_c | U, Z_D)` is complete, i.e. the only function `φ(U)` satisfying `E[φ(U) | Z_D, z_c] = 0` almost surely is `φ = 0`. This is the standard proximal-inference completeness condition [17, 18].
- **(A3) Bridge existence.** There exists a confounding bridge `h(z_d, u)` such that

  ```text
  E[M | Z_D, z_c] = ∫ h(z_d, u) p(u | z_d, z_c) du
  ```
  as in [18, Thm. 1].
- **(A4) Positivity.** `p(z_d | z_c^(k)) > δ > 0` for every context in the bank, for some `δ > 0`.
- **(A5) Representative bank.** The samples `{z_c^(k)}` are i.i.d. from the marginal `p(z_c)` (or, more weakly, their empirical distribution converges to it).

**Then** under (A1)–(A5),

```text
(1 / K) sum_k p(m | z_d, z_c^(k))  →  p(m | do(z_d))   almost surely as K → ∞,
```

at rate `O_P(K^{-1/2})` plus a bounded bridge-approximation bias controlled by the capacity of the decoder `f_seg`.

A proof sketch follows [18]: under (A1)–(A3) the bridge `h` implements proximal g-computation, so `E_{z_c ~ p(z_c)} p(m | z_d, z_c) = p(m | do(z_d))`; (A5) makes the sample average consistent for this expectation, and (A4) guards against division by vanishing densities when positivity fails.

#### 4.1.4 Assumptions are encoder-side claims, not training consequences

We treat (A1)–(A3) as *hypotheses on the encoder*, not as properties guaranteed by the training procedure. Concretely:

- (A1) is a modelling assumption that the context encoder only sees context-relevant variation. It is architecturally supported by independent `E_d` / `E_c` weights (see §2.5) and by the region-level disease supervision that biases `E_d` toward disease features, but it is not proved.
- (A2)–(A3) are the usual proximal-inference conditions; they require that the latent bank is rich enough for a nontrivial bridge to exist. Failure modes (rank deficiency, collapsed banks) produce non-unique bridges and must be diagnosed empirically.

#### 4.1.5 Honest critiques and retreat clause

We deliberately cite the negative results for deep latent-confounder methods, because our estimator inherits their structural risks:

- **CEVAE inconsistency under misspecification** [15]: a neural latent-confounder model can be asymptotically biased if the latent variable does not capture the true confounder structure.
- **Deconfounder critiques** [14, 16]: averaging predictions over learned multi-cause latents does not, in general, identify causal effects.

Specific failure modes the CRN must diagnose in evaluation:

- **Residual hidden confounding** if `z_c` is not a sufficient proxy for `U` — the dominant failure mode. Mitigated empirically by testing counterfactual invariance under acquisition interventions (§ experiments).
- **Selection bias in the bank** (Castro, Walker, and Glocker [19]) if the training cohort is acquisition-biased — addressed by out-of-distribution protocols (BraTS-Africa, FeTS, ROOD-MRI).
- **Positivity violations** on rare `(z_d, z_c)` combinations — mitigated by sampling the bank from the full training marginal, not per-batch only.
- **Descendant / post-treatment bias** if `z_c` ever becomes a function of `z_d`. This would convert `z_c` into a descendant of the treatment and reintroduce collider bias. Preventing it is why the two encoders are kept fully independent (§2.5, `src/crn/models.py:524-537`); any future architectural change sharing weights between the two encoders would invalidate the proposition.
- **Completeness failure** yielding non-unique bridge solutions — detectable by instability of `p(m | do(z_d))` under resampling of the bank.

**Retreat clause.** If (A1)–(A3) cannot be defended for a given deployment (e.g. because the two encoders share part of `x` they should not, or the bank is too narrow to satisfy completeness), we retreat to an *empirical-Bayes marginalization* interpretation: the estimator is then not a causal intervention but a context-robustified observational prediction, and the paper's claims weaken from identification of `p(m | do(z_d))` to context-robustness of `p(m | z_d, Z_c)`. This weaker claim is still experimentally meaningful — it is what the OOD protocols directly test — but the causal language in §4 would have to be revised accordingly.

#### 4.1.6 Related bank-style implementations

Our estimator generalizes a family of recent bank-based backdoor approximations that use *discrete* context banks — CONTA [20] for weakly-supervised segmentation, IFSL [21] for few-shot learning, and long-tailed classification [22] — to the *continuous learned* setting. The closest published analogue to bank-averaging in a deep model is Xu and Gretton [23], which implements backdoor adjustment via learned feature-level mean embeddings. Unlike those works, we acknowledge explicitly that moving from discrete observed contexts to continuous learned contexts requires the proximal reframing above, because only then does the identification argument survive the fact that the adjustment variable is itself a learned quantity.

---

**Proximal / backdoor-adjustment references.**
[11] J. Pearl. *Causality*, 2nd ed., Cambridge University Press, 2009.
[12] Y. Wang, D. M. Blei. *The Blessings of Multiple Causes.* JASA 114(528):1574–1596, 2019.
[13] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, M. Welling. *Causal Effect Inference with Deep Latent-Variable Models (CEVAE).* NeurIPS 2017. arXiv:1705.08821.
[14] A. D'Amour. *On Multi-Cause Approaches to Causal Inference with Unobserved Counfounding: Two Cautionary Failure Cases and a Promising Alternative.* arXiv:1902.10286, 2019.
[15] S. Rissanen, P. Marttinen. *A Critical Look at the Consistency of Causal Estimation with Deep Latent-Variable Models.* NeurIPS 2021.
[16] E. L. Ogburn, I. Shpitser, E. J. Tchetgen Tchetgen. *Comment on "Blessings of Multiple Causes".* JASA 114(528):1611–1615, 2019. arXiv:2001.06555.
[17] W. Miao, Z. Geng, E. J. Tchetgen Tchetgen. *Identifying Causal Effects with Proxy Variables of an Unmeasured Confounder.* Biometrika 105(4):987–993, 2018. DOI 10.1093/biomet/asy038. arXiv:1609.08816.
[18] E. J. Tchetgen Tchetgen, A. Ying, Y. Cui, X. Shi, W. Miao. *An Introduction to Proximal Causal Learning.* Statistical Science 39(3), 2024. arXiv:2009.10982.
[19] D. C. Castro, I. Walker, B. Glocker. *Causality Matters in Medical Imaging.* Nature Communications 11:3673, 2020. DOI 10.1038/s41467-020-17478-w.
[20] D. Zhang, H. Zhang, J. Tang, X.-S. Hua, Q. Sun. *Causal Intervention for Weakly-Supervised Semantic Segmentation (CONTA).* NeurIPS 2020. arXiv:2009.12547.
[21] Z. Yue, H. Zhang, Q. Sun, X.-S. Hua. *Interventional Few-Shot Learning.* NeurIPS 2020. arXiv:2009.13000.
[22] K. Tang, J. Huang, H. Zhang. *Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect.* NeurIPS 2020. arXiv:2009.12991.
[23] L. Xu, A. Gretton. *A Neural Mean Embedding Approach for Back-door and Front-door Adjustment.* arXiv:2210.06610, 2022.

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

## 7. What Is Novel and How It Relates to Prior Work

### 7.1 Contribution

The novelty of the framework does not lie in using a U-Net-style decoder alone. The novelty is the causal mechanism layered on top of segmentation, combined with identifiability-aware theory and quantitative counterfactual evaluation:

1. **Causally corrected problem formulation.** We frame the task as causal effect estimation under latent confounding, not confounder removal.
2. **Identifiability-grounded latent factorization.** `z_d` / `z_c` are anchored to iVAE [2], content/style block-identifiability [3], and paired-intervention identifiability [5, 6], not to unsupervised disentanglement (which Locatello [1] proved is impossible).
3. **Latent bank backdoor adjustment, reframed as proximal causal inference.** We make the dependence on proxy-validity (A1)–(A5) explicit and acknowledge the deconfounder/CEVAE critiques [14, 15, 16] rather than ignoring them.
4. **Matched counterfactual regularization with bounded context influence.** Context swaps stabilize predictions without assuming zero context effect.
5. **Disease-counterfactual supervision under fixed context.** Disease swaps approximate lesion interventions in a controlled latent setting.
6. **Lesion-aware counterfactual contrastive learning.** We align factual lesion features with adjusted same-disease features and separate them from hard disease-counterfactual lesion features.
7. **Output-image causal interpretability.** The framework produces counterfactual images and causal effect maps rather than relying only on post hoc attention.
8. **Quantitative counterfactual evaluation.** Context Invariance (M1), Disease-Swap Effect (M2), Adebayo sanity check (M3), and axiomatic soundness (M5: Composition / Reversibility / Effectiveness, after Monteiro et al. [31]) — counterfactuals are scored, not only rendered.
9. **Out-of-distribution payoff.** BraTS-Africa [28], ROOD-MRI / TorchIO corruptions [29, 30], and FeTS leave-one-institution-out [32, 33] directly test the context-robustness implied by (A1).

### 7.2 Related work

**Bank-based backdoor analogues (discrete).** CONTA [20], IFSL [21], and long-tailed causal classification [22] implement backdoor adjustment by averaging over a *discrete observed* context bank. Our estimator generalizes this paradigm to the *continuous learned* setting; §4.1 explains why that generalization requires the proximal reframing.

**Deep structural causal models for medical imaging.** Pawlowski, Castro, and Glocker [24] implement Pearl's three rungs of causation with normalizing flows on brain MRI, including null-intervention roundtrip checks. Castro, Walker, and Glocker [19] is the canonical reference for causal diagrams in medical imaging, including selection bias and acquisition confounding. Reinhold, Carass, and Prince [25] build a structural causal model for multiple sclerosis MRI and is the closest prior template for disease-covariate swap. Monteiro et al. [31] give the axiomatic framework (Composition / Reversibility / Effectiveness) that our M5 metric suite instantiates.

**Counterfactual invariance and spurious correlation.** Veitch, D'Amour, Yadlowsky, and Eisenstein [26] formalize counterfactual invariance of learned representations and distinguish causal from anti-causal structural models. Our bounded context-stability loss §4.2 implements a weak form of counterfactual invariance that is consistent with the causal SCM direction in that work.

**Domain generalization and causality-inspired augmentation.** GIN+IPA (Ouyang et al. [27]) is the key single-source domain-generalization baseline our OOD protocols must outperform. DANN (Kamnitsas et al. in medical imaging) and BigAug (Zhang et al.) are the non-causal baselines. Our claim is that bank-based proximal adjustment, at evaluation time, yields a robustness signature distinct from augmentation- or adversarial-training approaches.

**2024–2025 causal-BraTS competitors.** A small cluster of papers now applies causal tools to BraTS: Liu et al. [34] implement front-door adjustment on BraTS 2020/2021 but report in-distribution metrics only; SI²CRL [35] uses frequency-amplitude causal intervention for single-domain generalization; CF-Seg (Mehta et al. [36]) is the most direct competitor, computing Dice on factual vs. counterfactual images at MICCAI 2025. Our differentiator is the combination of (i) identifiability-grounded theory, (ii) proximal backdoor with explicit (A1)–(A5), (iii) axiomatic counterfactual evaluation, and (iv) cross-population + acquisition-corruption + cross-site OOD evaluation — no competitor combines all four.

Taken together, the contribution is a segmentation framework whose novelty is the *combination* of identifiability theory, proximal adjustment, axiomatic counterfactual evaluation, and OOD robustness on a medical task with built-in distribution shift.

---

**Additional references.**
[24] N. Pawlowski, D. C. Castro, B. Glocker. *Deep Structural Causal Models for Tractable Counterfactual Inference.* NeurIPS 2020. arXiv:2006.06485.
[25] J. C. Reinhold, A. Carass, J. L. Prince. *A Structural Causal Model for MR Images of Multiple Sclerosis.* MICCAI 2021. DOI 10.1007/978-3-030-87240-3_75.
[26] V. Veitch, A. D'Amour, S. Yadlowsky, J. Eisenstein. *Counterfactual Invariance to Spurious Correlations.* NeurIPS 2021. arXiv:2106.00545.
[27] C. Ouyang, C. Chen, S. Li, Z. Li, C. Qin, W. Bai, D. Rueckert. *Causality-Inspired Single-Source Domain Generalization for Medical Image Segmentation (GIN+IPA).* IEEE TMI 42(4):1095–1106, 2023. arXiv:2111.12525.
[28] M. Adewole, J. D. Rudie, A. Gbadamosi, et al. *The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population.* arXiv:2305.19369, 2023.
[29] L. Boone et al. *ROOD-MRI: Benchmarking the Robustness of Deep Learning Segmentation Models to Out-of-Distribution and Corrupted Data in MRI.* NeuroImage 278:120289, 2023. DOI 10.1016/j.neuroimage.2023.120289.
[30] F. Pérez-García, R. Sparks, S. Ourselin. *TorchIO: A Python Library for Efficient Loading, Preprocessing, Augmentation and Patch-based Sampling of Medical Images.* Computer Methods and Programs in Biomedicine 208:106236, 2021. arXiv:2003.04696.
[31] M. Monteiro, F. De Sousa Ribeiro, N. Pawlowski, D. C. Castro, B. Glocker. *Measuring Axiomatic Soundness of Counterfactual Image Models.* ICLR 2023. arXiv:2303.01274.
[32] S. Pati et al. *The Federated Tumor Segmentation (FeTS) Challenge.* arXiv:2105.05874, 2021.
[33] S. Pati et al. *Federated learning enables big data for rare cancer boundary detection.* Nature Communications 13:7346, 2022. DOI 10.1038/s41467-022-33407-5.
[34] M. Liu, Y. Li, X. Nie, Y. Xu, D. Liu. *Causal Intervention for Brain Tumor Segmentation.* MICCAI 2024. DOI 10.1007/978-3-031-72114-4_16.
[35] SI²CRL. *Single-domain generalization via frequency-amplitude causal intervention.* Medical Image Analysis, 2025. (S1361841525002889.)
[36] R. Mehta, F. De Sousa Ribeiro, T. Xia, F. Roschewitz, A. Santhirasekaram, A. Marshall, B. Glocker. *CF-Seg: Counterfactuals Meet Segmentation.* MICCAI 2025. arXiv:2506.16213.

## 8. Current Empirical Snapshot

In the current BraTS validation setting, the strongest version is the 2.5D causal-contrastive model. It keeps the same causal mechanism but replaces the planar backbone with a depth-aware slice-stack encoder/decoder. Compared with the previous best 2D causal-contrastive model, it improves volume-level segmentation:

```text
Best swept volume mean Dice: 0.7797 -> 0.7963
Best swept volume mean HD95: 11.25  -> 9.68
WT Dice: 0.8640 -> 0.8702
TC Dice: 0.7678 -> 0.7675
ET Dice: 0.7074 -> 0.7512
```

The gain is not only numerical. The causal counterfactual panels also show the desired qualitative behavior: context interventions usually induce only small output changes, while disease interventions produce larger lesion-relevant effects.

The continuation ablations also clarify which causal terms matter most:

- removing **region adjustment** causes the largest drop
- removing **region disease swap** causes the second largest drop
- removing the **lesion-aware contrastive** term causes a smaller but still measurable degradation, especially in HD95
- removing **region context stability** slightly changes Dice but worsens HD95, indicating that it mainly acts as a geometric regularizer

## 9. Limitations and Next Steps

The current framework is already causally stronger than the earlier confounder-removal version, but several directions remain open:

- tune `lambda_region_cf_stability` downward, since ablations suggest it helps HD95 more than Dice
- strengthen ET-focused disease interventions, because ET remains the hardest region
- tighten lesion-preservation constraints in counterfactual reconstructions
- expand the interpretability study into a paper-grade quantitative analysis
- move from the current 2.5D depth-aware backbone toward a stronger full 3D or nnU-Net-style backbone while preserving the same causal adjustment and counterfactual objectives

These are natural extensions of the same causal direction rather than departures from it.

## 10. Summary

Our framework proposes a causally grounded segmentation system in which disease and context are explicitly modeled as separate latent factors, prediction depends on both, causal effect estimation is approximated through adjustment, and explanations are produced through image-space counterfactual interventions. The central conceptual move is simple but important: under confounding, the right objective is not to remove confounders from prediction, but to adjust for them while preserving genuine causal structure.
