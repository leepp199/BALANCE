# Structure-Referenced Open-World Few-Shot Class-Incremental Audio Learning

## Abstract

We study open-world few-shot class-incremental audio learning, where a learner must
reject unknown sounds, discover their categories, and acquire them without erasing base
knowledge. We introduce a **Latent Structure Reference Bank (LSRB)** that participates in
base representation learning through a cross-sample structure-consistency objective. LSRB
provides a stable reference for novelty scoring and discovery in later mixed streams,
without interpreting late convolutional axes as physical time or frequency. To address
few-shot instability and catastrophic forgetting, we combine uncertainty-aware class
moments, covariance shrinkage, reliability-gated hard statistical replay, and bounded
residual prototype updates. Novel candidates are registered by **Capacity-Aware Novel-Class
Assignment (CANA)**. We evaluate LS-100, NS-100, and FSC-89 under repeated mixed-stream
protocols, comparing recent open-world and continual category-discovery methods under a
common audio feature extractor and reporting detection, discovery, incremental accuracy,
forgetting, and calibration metrics.

## Contributions

1. LSRB is trained jointly with the base encoder and then frozen as a cross-session latent
   reference; this makes the structural module part of representation learning rather than
   an offline clustering add-on.
2. CANA converts rejected samples into capacity-controlled novel-class assignments and
   prevents the mode-collapse behavior of unconstrained clustering.
3. Uncertainty is decomposed into predictive entropy, mutual information, and centered
   prediction dispersion; the resulting reliability gates control replay and prototype
   write-back.
4. The evaluation reports exact paired repeats on all three datasets and distinguishes
   original-protocol baselines from transferred SOTA implementations.

## Main comparison plan

The table will include the project classifier, TEEN, OFCL, OPCR, YLOC, Happy, CaMP,
MetaGCD, OCGCD, OpenIncrement, and VB-CGCD where their protocols can be transferred
without hidden labels. Main rows are: baseline; LSRB without structural loss; LSRB with
structural loss; LSRB+CANA; + uncertainty memory; + reliability-gated replay; and the
complete model. All values are inserted from the 50-repeat raw result ledger only.

## Figure plan

Figure 1: overview pipeline (base encoder → jointly trained LSRB → mixed-stream novelty
gate → CANA → prototype expansion/replay). Figure 2: uncertainty decomposition and
reliability gates. Figure 3: per-session accuracy/forgetting curves. Figure 4: OSR ROC and
candidate-discovery quality. Figure 5: ablation matrix for LSRB participation, CANA,
replay, and residual update.
