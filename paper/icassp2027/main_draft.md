# Structure-Referenced Open-World Few-Shot Class-Incremental Audio Learning

**Target venue:** ICASSP 2027  
**Status:** Experimental draft; numerical entries marked `TBD` are populated only from the 50-repeat result ledger.

## Abstract

Open-world few-shot class-incremental audio learning requires a learner to reject unknown
inputs, discover their latent categories, and acquire them without erasing previously
learned classes. Existing pipelines often separate open-set detection from incremental
classification, or interpret late convolutional axes as physical time-frequency factors,
which does not remain valid after repeated convolution, downsampling, and channel mixing.
We propose a structure-referenced framework built around a **Latent Structure Reference
Bank (LSRB)**. LSRB is learned jointly with the base encoder through a cross-sample
structure-consistency objective and is subsequently frozen as a common reference for
novelty scoring and discovery. Rejected samples are organized by **Capacity-Aware
Novel-Class Assignment (CANA)**, which prevents unconstrained clustering from collapsing
most candidates into one cluster. To reduce few-shot overfitting and catastrophic
forgetting, we combine uncertainty-aware class moments, covariance shrinkage,
reliability-gated hard statistical replay, and bounded residual prototype updates. We
evaluate LS-100, NS-100, and FSC-89 under repeated mixed known/novel streams and compare
recent open-world and continual category-discovery methods using detection, discovery,
incremental accuracy, forgetting, and calibration metrics.

## 1. Introduction

Few-shot class-incremental audio classification generally assumes labeled support examples
for every novel class. Few-shot open-set recognition, in contrast, stops after deciding
whether a query is unknown. A deployed audio learner must perform both operations
repeatedly: detect novelty, discover categories in an unlabeled stream, expand its
classifier, and retain earlier knowledge.

Two practical failures dominate this setting. First, an unknown stream is contaminated by
old classes and background variation, so an unconstrained clustering step can merge
several novel classes. Second, five-shot prototypes have high variance and can overwrite
or compete with reliable base prototypes, causing catastrophic forgetting.

Our framework addresses these failures with four linked design choices:

1. **LSRB training participation.** Cross-sample latent descriptors update a reference
   bank while a structure-consistency loss regularizes the encoder. LSRB is therefore part
   of representation learning, not an offline post-processing step.
2. **CANA discovery.** Rejected candidates are assigned to capacity-controlled emerging
   class prototypes using a global minimum-cost assignment.
3. **Uncertainty decomposition.** Predictive entropy, mutual information, and centered
   Monte-Carlo prediction dispersion are combined to estimate both confidence and
   predictive disagreement.
4. **Reliability-controlled adaptation.** Class moments, covariance shrinkage, hard replay,
   and bounded residual updates improve plasticity for new classes while limiting drift of
   old classes.

## 2. Related Work

### 2.1 Open-set and open-world recognition

OpenMax introduced post-hoc open-set calibration for neural classifiers. OpenIncrement
formulated open-world incremental recognition with explicit unknown handling. MetaGCD and
OCGCD studied generalized and online category discovery, respectively. These methods
motivate our mixed-stream protocol, but their original experiments are predominantly
visual and do not jointly evaluate audio open-set rejection, unlabeled novel discovery,
and few-shot class registration.

### 2.2 Few-shot class-incremental learning

TEEN provides training-free prototype calibration for few-shot incremental sessions.
OFCL, OPCR, and YLOC address classifier or representation stability, while recent methods
such as Happy, CaMP, and VB-CGCD target continual category-discovery bias, projected
distillation, or covariance alignment. We transfer these methods only when their required
labels and feature protocol can be reproduced; transferred results are explicitly marked.

### 2.3 Uncertainty and prediction dispersion

Monte-Carlo dropout provides epistemic uncertainty estimates. Prediction-matrix analyses
show that nuclear norm can characterize confidence and dispersity, but an uncentered norm
is not a monotonic uncertainty score. We therefore use the nuclear norm only on the
centered prediction matrix as a disagreement component, together with entropy and mutual
information.

## 3. Proposed Method

### 3.1 Problem formulation

Let session zero contain labeled base classes \(\mathcal C_0\). Session \(t\) provides an
unlabeled mixed stream containing registered classes and \(K\) novel classes with only a
few samples available for registration after discovery. The model must output either a
registered class or `unknown`, group unknown samples into novel categories, and update the
classifier for future sessions.

### 3.2 Latent Structure Reference Bank (LSRB)

A ResNet-18 encoder produces a pooled embedding \(z\in\mathbb R^{512}\) and local
layer-4 descriptors \(f_i\). Normalized descriptors from different base samples are
reservoir-sampled into a shared bank \(B=\{b_k\}_{k=1}^{M}\). The structural response is

\[
a_{ik}=\frac{\exp(\cos(f_i,b_k)/\tau)}{\sum_j\exp(\cos(f_i,b_j)/\tau)},\qquad
h_k(x)=\frac{1}{HW}\sum_i a_{ik}.
\]

Unlike an interpretation of feature axes as time or frequency, LSRB centers are recurring
references in the task-adapted latent space. During base training, the bank response is
fed back to the encoder through

\[
\mathcal L_{\mathrm{LSRB}}=\frac{1}{M}\sum_k
\|\operatorname{sg}(\bar h_k)-h_k(x)\|_2^2,
\]

where \(\bar h_k\) is the batch reference response and `sg` denotes stop-gradient on the
target branch. The base objective is

\[
\mathcal L_{\mathrm{base}}=\mathcal L_{\mathrm{CE}}+
\lambda_s\mathcal L_{\mathrm{LSRB}}+\lambda_c\mathcal L_{\mathrm{center}}.
\]

After base convergence, LSRB is frozen for cross-session evaluation. Historical code
aliases `DFSB` and `balanced_kmeans/BCD` are retained only for reproducibility and refer
to LSRB and CANA, respectively.

### 3.3 Open-world detection and CANA

Registered samples are classified by similarity to reliability-weighted prototypes.
Candidates below the novelty threshold enter discovery. We initialize candidate centers
in normalized semantic-structural space and solve a capacity-constrained assignment:

\[
\min_A\sum_{ik} A_{ik}\|g_i-c_k\|_2^2,
\quad \sum_k A_{ik}=1,
\quad \sum_i A_{ik}\in\{\lfloor N/K\rfloor,\lceil N/K\rceil\}.
\]

Expanding each center into capacity slots permits a Hungarian assignment. CANA uses the
known stream construction but no candidate labels, reducing cluster collapse while
preserving open-world operation.

### 3.4 Uncertainty-aware memory and anti-forgetting update

For each registered class, memory stores \(M_c=(\mu_c,v_c,n_c,\rho_c)\): normalized mean,
diagonal variance, count, and reliability. Novel covariance is shrunk toward related old
classes:

\[
\widetilde v_n=(1-\eta_n)\widehat v_n+
\eta_n\sum_c w_{nc}v_c,
\quad w_{nc}=\operatorname{softmax}(\cos(\widehat\mu_n,\mu_c)/\tau_m).
\]

Uncertainty combines predictive entropy, mutual information, and the nuclear norm of the
centered Monte-Carlo prediction matrix. High disagreement lowers reliability. Synthetic
hard pairs sampled from class moments are replayed only when the base accuracy and variance
gates are valid. Prototype optimization uses replay cross-entropy and a reliability anchor:

\[
\mathcal L=\sum_i q_{y_i}\operatorname{CE}(\bar x_i,W)+
\lambda_a\sum_c(2-\rho_c)\|w_c-w_c^{old}\|_2^2.
\]

The write-back is bounded by \(W\leftarrow(1-\delta)W^{old}+\delta W^{rep}\), preventing
synthetic errors from overwriting stable old prototypes.

## 4. Reproduction Protocol and Main Experiments

We first reproduce the original project test form on LS-100, NS-100, and FSC-89. LS-100
and NS-100 use 80 base classes and FSC-89 uses 69. Each benchmark contains four
five-way, five-shot incremental sessions. The reproduced protocol reports the base
session accuracy, each incremental-session accuracy, the average incremental accuracy,
the final all-seen accuracy, and forgetting. We then run the normal mixed open-world
stream, in which registered and novel samples arrive together and the model must reject,
discover, and register candidates online. All methods use the same stream construction,
feature extractor, classifier reset policy, and random seeds. The final reported values
use 50 independently reset repeats with paired streams.

Metrics include known/unknown recall, F1, AUROC, AUPR, FPR95, clustering ACC/NMI/ARI,
incremental-only accuracy, old-class accuracy, all-seen accuracy, final accuracy, and
forgetting. All methods use the same audio preprocessing and frozen feature interface when
transferred from visual protocols.

## 5. Main Results: Reproduction and SOTA Comparison

The main table contains the reproduced project baseline and original method variants,
followed by TEEN, OFCL, OPCR, YLOC, Happy, CaMP, MetaGCD, OCGCD, OpenIncrement, and
VB-CGCD. Each entry is labeled as exact reproduction, official transfer, or
protocol-compatible reimplementation. This table is reserved for end-to-end method
comparisons; no component ablation is mixed into it.

Following the reporting style of FSCIL and continual-GCD papers, the main classification
table reports accuracy after every session rather than only one terminal number. We also
report the harmonic mean of old-class and new-class accuracy (HMean) and the mean novel
class accuracy (NAcc), which expose the old/new trade-off hidden by all-seen accuracy.
For discovery, ACC, NMI, and ARI are reported separately from classification accuracy.

### 5.1 Reproduced original test form

| Method / protocol | LS-100 S0/S1/S2/S3/S4 | NS-100 S0/S1/S2/S3/S4 | FSC-89 S0/S1/S2/S3/S4 | Avg. inc. | Final | Forgetting |
|---|---|---|---|---:|---:|---:|
| Original project baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| Original project method, reproduced | TBD | TBD | TBD | TBD | TBD | TBD |
| LSRB-trained replacement, same test form | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.2 Normal mixed open-world stream

| Method | LS-100 Avg/HMean/NAcc | NS-100 Avg/HMean/NAcc | FSC-89 Avg/HMean/NAcc | AUROC | AUPR | FPR95 | ACC/NMI/ARI |
|---|---|---|---|---:|---:|---:|---|
| Project baseline | TBD | TBD | TBD | TBD | TBD | TBD |
| TEEN | TBD | TBD | TBD | TBD | TBD | TBD |
| OFCL / OPCR / YLOC | TBD | TBD | TBD | TBD | TBD | TBD |
| Happy / CaMP | TBD | TBD | TBD | TBD | TBD | TBD |
| MetaGCD / OCGCD / OpenIncrement | TBD | TBD | TBD | TBD | TBD | TBD |
| VB-CGCD | TBD | TBD | TBD | TBD | TBD | TBD |
| Full LSRB + CANA model | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.3 Calibration and open-set analysis

Following open-set recognition practice, we include ROC and precision-recall curves, not
only scalar AUROC. The table reports known recall, unknown recall, F1, AUROC, AUPR, and
FPR95 at the operating threshold selected on the validation split. Threshold-free results
are shown separately from incremental classification results.

No value is reported from a single screening run; all final cells come from the 50-repeat
ledger.

**Current training checkpoint (not a final table cell).** The NS-100 repair run has
completed its 15-epoch meta-training pass. Its single diagnostic evaluation reports
known accuracy 0.7206, unknown accuracy 0.8065, AUROC 0.9356, F1 0.7718, average
incremental accuracy 0.5332, final all-seen accuracy 0.9383, and PD=6.01%. These
numbers are retained as an engineering checkpoint only; they are excluded from the
main table until the original test form and the paired 50-repeat mixed-stream ledger
are complete.

## 6. Ablation Experiments

The ablation table is separate from the main comparison. Every row uses the same reproduced
mixed-stream protocol and reports component-level changes relative to the full model.

| Variant | LSRB trained jointly | CANA | Uncertainty memory | Reliability gate | Bounded update | Avg. inc. | Forgetting |
|---|:---:|:---:|:---:|:---:|:---:|---:|---:|
| Baseline | no | no | no | no | no | TBD | TBD |
| LSRB, offline only | no | no | no | no | no | TBD | TBD |
| LSRB joint training | yes | no | no | no | no | TBD | TBD |
| LSRB + CANA | yes | yes | no | no | no | TBD | TBD |
| + uncertainty memory | yes | yes | yes | no | no | TBD | TBD |
| + reliability-gated replay | yes | yes | yes | yes | no | TBD | TBD |
| Full model | yes | yes | yes | yes | yes | TBD | TBD |

We additionally report base train/test gaps to quantify overfitting, old-class retention and
forgetting to quantify stability, and novel-class accuracy to quantify plasticity.
Sensitivity studies vary bank size, \(\lambda_s\), covariance shrinkage, replay strength,
and residual step \(\delta\).

## 7. Visualization Plan

**Figure 1 — Task and method overview:** follow the continual-GCD setting diagrams: labeled
base training, then sequential unlabeled mixed streams, with known samples retained and
novel samples discovered. Overlay the LSRB training loop and CANA assignment.

**Figure 2 — Per-session performance:** plot all-seen, old-class, and novel-class accuracy
for S0--S4, plus a second panel for forgetting. This mirrors FSCIL papers that show the
session trajectory rather than only the final average.

**Figure 3 — Open-set detection:** ROC and precision-recall curves for the three datasets,
with AUROC/FPR95 in the legend or a companion table.

**Figure 4 — Category discovery:** ACC, NMI, and ARI by session, with candidate purity and
novel-class coverage as diagnostic curves. A two-dimensional embedding visualization is
included only as qualitative evidence, never as the primary result.

**Figure 5 — Component ablation:** heatmap or grouped bars for the separate ablation table:
LSRB joint training, CANA, uncertainty memory, reliability gates, and bounded update.

**Figure 6 — Efficiency and stability:** memory footprint, update time, and prototype drift
versus the baseline. This is important because LSRB is a persistent reference bank and
the replay mechanism is intended to remain exemplar-free.

The framework figure uses a flat vector style with three functional color groups: base
training, open-world discovery, and anti-forgetting adaptation. It avoids decorative
gradients and avoids any time/frequency-axis interpretation of the latent descriptors.

## 8. Limitations and Reproducibility

The stream construction assumes a known number of emerging classes per session; this is
used only as a capacity prior, not as a candidate label. Image-oriented baselines may not
transfer perfectly to audio, so their provenance is reported. Checkpoints, stream seeds,
per-session records, and the 50-repeat aggregation script will be released with the
submission package.

## 9. Conclusion

LSRB turns cross-sample latent structure into a trainable and reusable reference for
open-world audio learning. CANA controls novel-class allocation, while uncertainty-aware
memory and bounded replay address the complementary problems of few-shot overfitting and
catastrophic forgetting. The final ICASSP submission will report the complete three-dataset
comparison only after the 50-repeat evaluation and all planned ablations are complete.

## References

The citation list is maintained in [`references.bib`](./references.bib) and includes
OpenMax, OpenIncrement, MetaGCD, OCGCD, TEEN, Happy, CaMP, VB-CGCD, uncertainty estimation,
prediction-dispersion analysis, and audio few-shot class-incremental learning methods.
The reporting choices follow the per-session FSCIL accuracy and HMean/NAcc convention of
TEEN, the mixed known/novel continual-stream protocol of MetaGCD, and the online stream
and discovery diagnostics emphasized by OCGCD.
