# Structure-Referenced Open-World Few-Shot Class-Incremental Audio Learning

> Working ICASSP manuscript. Values marked **pending** must be filled only by the
> collectors after the corresponding repeated run completes.
>
> **Raw-v2 correction:** legacy TEEN/FOWAC repeated-run aggregates in the working notes
> below are withdrawn because the evaluator averaged cumulative means. They are kept
> temporarily for provenance, not as manuscript evidence. Corrected paired ten-stream
> runs are active and all final tables/statistics will be regenerated from explicit
> `[RAW]` round/session records. Independent feature-transfer baseline rows are unaffected.

## Abstract

Open-world class-incremental audio learning requires a model to separate previously
seen and novel inputs, discover latent novel categories from a sparse unlabeled stream,
and absorb them without erasing earlier knowledge. Existing audio few-shot incremental
methods generally assume labeled novel support, whereas few-shot open-set methods stop
after rejection. We study their joint setting under mixed known/novel streams and propose
a structure-referenced continual prototype framework. A cross-sample Deep Feature
Latent Structure Reference Bank (LSRB; implementation alias DFSB) is learned from task-adapted layer-4 descriptors and retained as a
common reference for representation learning, novelty scoring and class discovery. To
stabilize continual expansion, an exemplar-free uncertainty memory stores class moments
and reliability, calibrates five-shot novel covariance, and performs confusion-aware
statistical replay with confidence anchoring and residual prototype updates. Experiments
use LS-100, NS-100 and the environmental-sound FSC-89 benchmark under a common
base-plus-four-session mixed-stream protocol. We report known/unknown discrimination,
clustering, incremental-only and all-seen performance with repeated streams and explicit
result provenance. **Final comparative numbers pending completion of repeated runs.**

## 1. Introduction

Few-shot class-incremental audio classification (FCAC) learns new audio categories from
limited support while retaining base knowledge, but conventionally assumes that every
incoming item belongs to a labeled novel class. Few-shot open-set recognition (FSOR), by
contrast, separates known and unknown queries but does not organize unknown samples into
new semantic classes. A deployed system must perform both operations repeatedly: detect
novelty in a mixed unlabeled stream, discover categories, expand its classifier, and use
the expanded knowledge in the next session.

The earlier formulation attempted to address representation scarcity through
sample-wise spatial clustering and interpreted deep ResNet axes as physical time and
frequency. This interpretation is not technically justified after repeated convolution,
downsampling and channel mixing, and sample-specific cluster identities cannot serve as
a stable reference across sessions. We replace it with a dataset-level structure view:
task-adapted local descriptors from different base samples are clustered once to learn a
shared LSRB. Its centers are recurring references in latent feature space, not phonetic or
acoustic units, and their number is unrelated to the number of semantic classes.

Continual prototype expansion introduces a second problem. Freezing the encoder protects
base representations but does not prevent classifier interference: noisy five-shot novel
centers increasingly compete with old prototypes. We therefore store class-level moments
instead of waveforms, shrink uncertain novel covariance toward related old distributions,
and replay statistically generated hard pairs. A confidence-weighted anchor and a bounded
residual update prevent synthetic errors from overwriting observed prototypes.

Our contributions are:

1. We formulate a reproducible mixed-stream open-world FCAC protocol covering speech,
   musical instruments and environmental sounds, with end-to-end detection, discovery and
   continual classification metrics.
2. We replace sample-specific axis clustering with an LSRB learned once during
   the base phase and reused for structural response, novelty estimation, joint discovery
   and dual-prototype expansion.
3. We introduce exemplar-free uncertainty memory and confidence-anchored hard statistical
   replay to improve novel-class estimation and reduce old/new classifier interference.
4. We provide three-dataset comparisons with recent official transfers and paper-faithful
   reimplementations, while separating exact, transferred, failed and incompatible results.

## 2. Related Work

### 2.1 Few-shot class-incremental audio classification

FCAC methods typically freeze a base-trained embedding extractor and continually expand a
prototype, stochastic or regression classifier. Recent audio work studies prototype
adaptation, stability/plasticity balancing, fully few-shot base regimes and multi-level
audio embeddings. These methods inform our classifier baselines but do not themselves
solve unknown rejection and unlabeled category discovery.

### 2.2 Few-shot open-set recognition

FSOR methods model unknown space through extreme-value tails, task-adaptive negative
prototypes or diversified negative-prototype generators. They provide relevant rejection
components, but episodic FSOR terminates after assigning an unknown decision. Our protocol
places each detector before a fixed discovery and expansion stage to measure end-to-end
consequences.

### 2.3 Generalized and continual category discovery

GCD assumes unlabeled data containing old and novel categories, while CGCD repeats this
process over sessions and must retain previous knowledge. Recent approaches include Happy's
hardness-aware prototype augmentation and group-wise entropy regularization (NeurIPS 2024),
CAMP's projected distillation and category adaptation (ECCV 2024), PromptCCD's Gaussian
mixture prompt pool (ECCV 2024), covariance-aware VB-CGCD (ICML 2025), and the Fix-and-Explore
stability/plasticity strategy (AAAI 2026). TEEN, VB-CGCD, OFCL and OPCR represent semantic
calibration, Bayesian distribution modeling, metric-ball open learning and orthogonal/
confidence refinement respectively. Since most published settings use image encoders and
many labeled samples per session, we separate official-code runs, common-feature component
transfers and adjacent-task comparisons instead of transplanting paper-reported numbers.

## 3. Problem Formulation

Session 0 provides labeled base data
\(\mathcal D_0=\{(x_i,y_i)\}\) over \(C_0\) classes. At incremental session \(t\), an
unlabeled stream \(\mathcal U_t\) mixes samples from already registered classes and
\(N=5\) novel classes, with five observations per sampled class. The learner predicts a
known/unknown decision, clusters rejected candidates, registers novel classes and is then
evaluated over every class seen through session \(t\). No previous waveform is retained.

## 4. Method

### 4.1 Base warm-up and shared deep-feature structure

A ResNet-18 encoder produces a layer-4 map
\(F(x)\in\mathbb R^{512\times H\times W}\) and pooled semantic embedding
\(z(x)\in\mathbb R^{512}\). After supervised warm-up, normalized descriptors from all
base samples are reservoir sampled and fitted by MiniBatch K-means, yielding
\(B=\{b_k\}_{k=1}^{K}\). The bank is fixed during open-world sessions.

For descriptor \(f_i\), the soft assignment and sample structural response are

\[
a_{ik}=\frac{\exp(\cos(f_i,b_k)/\tau)}{\sum_j\exp(\cos(f_i,b_j)/\tau)},\qquad
h_k(x)=\frac{1}{HW}\sum_i a_{ik}.
\]

Masked structure prediction supplies an auxiliary base objective
\(\mathcal L_{base}=\mathcal L_{cls}+\lambda_{str}\mathcal L_{str}\). The axes are treated
only as latent spatial topology inherited from the input; no physical time/frequency
invariance is assumed.

### 4.2 Structure-referenced detection and discovery

Each registered class stores semantic prototype \(p_c\) and structural prototype \(r_c\).
The two similarities and bank quantization residual are

\[
s_c^{sem}=\cos(z,p_c),\quad s_c^{str}=\cos(h,r_c),\quad
e_{str}=\frac1{HW}\sum_i\min_k[1-\cos(f_i,b_k)].
\]

These quantities augment, rather than silently replace, the learned positive/negative
prototype boundary. Rejected samples are clustered using
\(g(x)=[\sqrt\alpha\,\bar z;\sqrt{1-\alpha}\,\bar h]\). A discovered class expands both
prototype banks, so base and incremental classes use the same representation.

### 4.3 Capacity-constrained balanced discovery

Sparse OSR candidates make ordinary K-means unstable: a dominant acoustic mode can
absorb most samples while other centers represent singletons, even when the stream is
approximately class balanced. We initialize centers with K-means and globally assign
candidate (i) to a center-slot using

\[
\min_{A\in\{0,1\}^{N\times K}}\sum_{i,k}A_{ik}\lVert g_i-c_k\rVert_2^2,
\quad \sum_k A_{ik}=1,
\quad \sum_i A_{ik}\in\{\lfloor N/K\rfloor,\lceil N/K\rceil\}.
\]

We expand each center into the required number of slots and solve the resulting
minimum-cost bipartite assignment with the Hungarian algorithm, then recompute centers.
Five assignment/centroid iterations are sufficient in our experiments. The constraint
uses only the known (K)-way session construction, not class labels, and remains defined
when OSR misses or leaks a few samples because cluster capacities differ by at most one.

### 4.4 Uncertainty memory and covariance calibration

For each class, memory \(M_c=(\mu_c,v_c,n_c,\rho_c)\) stores its normalized mean,
diagonal variance, count and reliability. A novel empirical variance \(\hat v_n\) estimated
from sparse candidates is shrunk toward similarity-weighted old variance:

\[
\tilde v_n=(1-\eta_n)\hat v_n+\eta_n\sum_c w_{nc}v_c,\quad
w_{nc}=\operatorname{softmax}(\cos(\hat\mu_n,\mu_c)/\tau_m),
\]

where \(\eta_n\) increases as sample reliability decreases.

### 4.5 Confidence-anchored hard statistical replay

Normalized pseudo-features are drawn from the stored diagonal distributions. Confusion
hardness is the maximum cosine similarity to a competing prototype and determines replay
weight. Joint prototype optimization minimizes weighted replay cross-entropy plus a
reliability-scaled anchor:

\[
\mathcal L_{rep}=\sum_i q_{y_i}\,CE(\bar x_i,W)+
\lambda_a\sum_c(2-\rho_c)\lVert w_c-w_c^{old}\rVert_2^2.
\]

The optimized bank is written back residually,
\(W\leftarrow(1-\delta)W^{old}+\delta W^{rep}\), with \(\delta=0.1\) in the screened
configuration. This bound is essential: unrestricted write-back is retained as a negative
ablation because it collapses sparse novel prototypes.

## 5. Experiments

### 5.1 Datasets and protocol

LS-100 and NS-100 use 80 base classes plus four 5-way sessions. FSC-89 uses 69 base
classes plus four 5-way sessions and supplies the environmental-sound setting requested by
reviewers. Every stream mixes five old and five novel classes with five samples per class.
Main results use ten repeats with classifier reset and report mean and standard deviation.

### 5.2 Metrics

We report base accuracy; known recall; unknown recall; F1; AUROC, AUPR and FPR95;
clustering ACC, NMI and ARI; incremental-only, old-class and all-seen accuracy; average
accuracy; final-session accuracy; performance degradation and forgetting. Runtime, peak
memory and result provenance accompany the main table.

### 5.3 Baselines and provenance

The comparison set contains the project frozen-prototype and DFSB variants, TEEN,
VB-CGCD, OFCL, OPCR and YLOC. Reviewer-named OAFN and fully-FCAC methods are listed
in a compatibility table because their noisy episodic or few-shot-base tasks differ from
the main protocol. DNPG is evaluated as a negative-prototype detector inside the common
discovery/expansion pipeline. “PCLAE-CTPN” remains uncited until a verifiable bibliographic
record is found.

### 5.4 Implementation and reproducibility

All waveforms are processed at 16 kHz. We use a 25-ms Hann window (400 samples),
a 10-ms hop (160 samples), and 128 log-Mel bins spanning 0--8 kHz. The encoder is
an ImageNet-initialized ResNet-18 adapted to the single-channel spectrogram input;
global pooling produces a 512-dimensional embedding and classification uses cosine
similarity. Base training uses SGD with Nesterov momentum 0.9, weight decay
$5\times10^{-4}$, batch size 128 and initial learning rate 0.005. The configured
schedule contains 30 supervised epochs followed by 15 episodic meta-training epochs;
each episode is 5-way 5-shot with 15 queries per class and a matched 5-way open set.

The continual evaluation contains session 0 and four 5-way 5-shot sessions. Each
incremental stream mixes five registered and five novel classes, five samples per
class. We repeat the complete sampled stream ten times, reset classifier state before
every repeat, and use paired streams for method comparisons. Unless otherwise stated,
TEEN calibration uses $\alpha=0.9$, all old prototypes, and temperature 0.0625.
Uncertainty replay uses diagonal covariance shrinkage 0.7, 16 pseudo-features per
class, 30 SGD steps at learning rate 0.03, anchor weight 2.0 and residual strength
0.1. Reliability gates disable replay when base accuracy is below 0.7 or mean
normalized variance is below $10^{-4}$. Experiments run on NVIDIA RTX 3090 GPUs;
software versions, exact commands, checkpoints and per-repeat records are retained
with the supplementary artifact.

### 5.5 Current verified checkpoints

- Canonical TEEN on LS-100: AA-inc 69.26%, final-inc 55.96% ± 2.98%, AA-all 89.39%.
- TEEN on NS-100: AA-inc 51.45%, final-inc 49.60% ± 1.12%, AA-all 93.44%.
- TEEN on FSC-89: AA-inc 24.23%, final-inc 16.33% ± 0.64%, AA-all 37.52%.
- Frozen-feature recent transfers (10 algorithm seeds each) show strong protocol
  dependence. OFCL/YLOC reach 59.86%/60.17% AA-inc on NS-100 but only 0.73%/0.00%
  on FSC-89; OPCR reaches 43.27%, 43.13%, and 1.02% on LS/NS/FSC; the Happy
  hardness-aware prototype component reaches 37.46%, 28.34%, and 5.56%.
  These are labeled acoustic component transfers rather than original image-protocol
  reproductions.
- Proposed cached-variance replay on LS-100 (10 paired streams): AA-inc 70.62%,
  final-inc 56.69% ± 2.55%, and AA-all 89.64%. Relative to canonical TEEN, the paired
  gains are +1.36 points (95% CI [0.86, 1.86], paired t-test p=0.000737), +0.72 points
  (p=0.0444), and +0.24 points (p=0.000230), respectively.
- DFSB dual-space discovery on LS-100 (weight 0.5; 10 independently reset mixed
  streams): AA-inc 71.78%, final-inc 60.82% $\pm$ 1.61%, and AA-all 89.76%.
  Relative to canonical TEEN, gains are +2.52 points (95% CI [1.73, 3.15],
  paired $p=0.000117$), +4.86 points ($p=6.34\times10^{-6}$), and +0.37 points
  ($p=2.85\times10^{-5}$), respectively. This run uses DFSB only for the discovery
  embedding and does not claim evidence for the combined DFSB+UMR model.
- Capacity-constrained balanced discovery on FSC-89 (10 paired streams) improves
  AA-inc from 24.23% to 26.38% (+2.15 points, 95% CI [1.62, 2.70], paired
  $p=4.56\times10^{-5}$), final-inc from 16.33% to 17.09% ($p=0.0128$), and
  AA-all from 37.52% to 37.81% ($p=5.28\times10^{-6}$). It also raises AUROC
  by 5.90 points and F1 by 5.48 points, both significant, although mean candidate
  discovery ACC itself is not uniformly higher. This supports balanced allocation as
  a downstream classifier stabilizer rather than a blanket clustering-quality claim.
- On NS-100 and FSC-89, the UMR reliability/variance gates are transparent: all ten
  paired metrics exactly equal TEEN. Ungated NS replay and forced FSC replay are retained
  as negative ablations, demonstrating that replay should be conditional rather than universal.
- The earlier one-stream 76.73% result is retained only as a screening observation and
  is not used as main-table evidence.

## 6. Required Ablations and Analysis

The final paper will compare: baseline; LSRB without training participation; LSRB with
structure-aware supervision; + structural detection; + joint clustering; + dual
prototypes; + uncertainty memory; + uniform replay; + hard replay; + confidence
anchor/residual update. Sensitivity covers LSRB size
\(K\in\{32,64,128,256\}\), structure loss, covariance shrinkage and residual strength.
Figures include per-session accuracy/forgetting, OSR ROC, clustering quality, prototype
drift, confusion matrices and memory/runtime trade-offs.

## 7. Conclusion

We study continual detection, discovery and expansion in sparse unlabeled audio streams.
The shared deep-feature structure avoids unsupported physical-axis assumptions and provides
a persistent cross-session reference, while uncertainty memory and bounded hard replay
address classifier interference without waveform storage. Final claims will be restricted to
the completed three-dataset repeated experiments.
