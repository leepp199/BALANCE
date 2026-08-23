# Baseline Survey for Few-Shot Open-World Audio Classification

## 1. Motivation

The target task requires a model to recognize previously learned audio classes, detect unknown audio samples, discover emerging classes, and incrementally update the classifier. Therefore, suitable baselines should not be limited to conventional few-shot classification. They should cover three related directions:

1. Few-shot class-incremental learning methods, which provide mechanisms for learning new classes with few samples while preserving old-class performance.
2. Open-set recognition or few-shot open-set recognition methods, which provide unknown rejection scores or open-set decision boundaries.
3. Open-world continual learning methods, which directly combine novelty detection, unknown discovery, and continual learning.

The uploaded OWCL paper is especially important because it theoretically connects OOD detection and class-incremental learning. It decomposes CIL into within-task prediction and task-id prediction, and argues that strong OOD detection is necessary for successful open-world continual learning. This supports the experimental design of combining incremental learning methods with open-set recognition methods. The uploaded OAFN paper is also relevant because it is an audio-specific open-world few-shot learning method, although it focuses mainly on noisy few-shot audio event classification rather than session-wise open-world continual learning.

---

## 2. Candidate Incremental Learning Methods

### 2.1 PITEL-CUSC / Pseudo-Incremental Stochastic Classifier, TASLP 2025

This is an audio-specific few-shot class-incremental audio classification method. It uses pseudo-incremental training for the embedding learner and a continually updated stochastic classifier. It is highly relevant because its task setting is close to audio FCAC. It can be used as a strong audio incremental baseline.

**Use in our experiments:**
Use its embedding learner and stochastic classifier as the incremental learning component. In each online session, new discovered classes are added through the stochastic classifier update. Unknown rejection is not native to this method, so it should be combined with an OSR score such as MLS, COSTARR, FOAC-AIFP-style open-set prototype score, or our adapted dual-prototype score.

**Strength:** audio-specific, close to FCAC.
**Limitation:** assumes labeled incremental samples and does not solve unknown discovery by itself.

---

### 2.2 Fully Few-Shot Class-Incremental Audio Classification, 2025

This method studies a more realistic audio setting where both base classes and incremental classes have only few samples. It uses a multi-level embedding extractor and a ridge regression classifier. It is useful for testing whether our method is better under stricter few-shot assumptions.

**Use in our experiments:**
Use the multi-level embedding extractor and ridge regression classifier as the incremental learner. For open-world evaluation, rejected unknown samples can be clustered and then passed to the ridge classifier as newly discovered classes.

**Strength:** audio-specific and fully few-shot.
**Limitation:** still mainly assumes labeled session data; open-set detection must be added externally.

---

### 2.3 Tri-WE, CVPR 2025

Tri-WE is a recent few-shot class-incremental learning method. It uses a tripartite weight-space ensemble involving the base model, previous model, and current model to balance stability and plasticity. It is not an audio method, but it is a strong FSCIL baseline and can be adapted to audio embeddings.

**Use in our experiments:**
Use the same audio encoder for fairness, then apply Tri-WE-style classifier/head updating across sessions. New-class weights are updated by weight-space interpolation and distillation. Combine it with an OSR method for unknown detection.

**Strength:** strong FSCIL baseline, explicitly addresses forgetting and overfitting.
**Limitation:** visual FSCIL method; audio adaptation needs careful implementation.

---

### 2.4 MACIL, ICML 2025 / arXiv 2025

MACIL targets task-agnostic class-incremental learning and addresses semantic drift through mean-shift compensation and covariance calibration. It is useful because open-world audio learning also suffers from feature drift when new classes are discovered and inserted into the classifier.

**Use in our experiments:**
Use MACIL-style mean and covariance calibration to update class prototypes or classifier weights after each session. Combine with open-set detectors for unknown sample selection.

**Strength:** strong task-agnostic CIL idea; useful for prototype-space drift correction.
**Limitation:** not specifically few-shot audio; may require simplification when only few samples are available.

---

### 2.5 SEC-Prompt, CVPR 2025

SEC-Prompt is a prompt-based FSCIL method that learns semantic complementary prompts, including discriminative prompts and non-discriminative prompts. It is suitable if we want a vision-language or audio-language baseline, for example using CLAP-like audio-text embeddings.

**Use in our experiments:**
If using CLAP/AudioCLIP features, replace image prompts with audio-text class prompts. Use prompt adaptation for incremental class learning. Unknown detection can be combined with text-based open-set scoring.

**Strength:** recent CVPR FSCIL method; useful for prompt-based comparison.
**Limitation:** originally visual; audio adaptation is nontrivial.

---

### 2.6 SDC, arXiv 2026

Static-Dynamic Collaboration divides FSCIL into a static retaining stage and a dynamic learning stage. The static part preserves old knowledge, while the dynamic part adapts to new classes. This is conceptually close to our encoder/classifier separation.

**Use in our experiments:**
Use the static branch as the frozen base encoder or base prototype memory, and use the dynamic branch to learn discovered new classes. Combine with OSR scoring for unknown rejection.

**Strength:** latest FSCIL direction; naturally matches stability-plasticity analysis.
**Limitation:** 2026 preprint; publication status should be verified before formal use.

---

## 3. Candidate Open-Set Recognition Methods

### 3.1 COSTARR, ICCV 2025

COSTARR is a recent open-set recognition method based on the attenuation hypothesis. It argues that small weights learned during training can attenuate features useful for distinguishing known and unknown classes. It consolidates information before and after attenuation for robust open-set recognition.

**Use in our experiments:**
Use COSTARR as the open-set detector on top of the audio embedding/classifier output. For each session, known samples are classified by the incremental classifier, while unknown samples are rejected according to the COSTARR score.

**Strength:** strong recent OSR baseline.
**Limitation:** visual OSR method; audio adaptation should be reported clearly.

---

### 3.2 UTL, ICCV 2025

Unknown Text Learning is a CLIP-based few-shot open-set recognition method. It learns unknown textual representations to improve unknown detection when unknown samples and unknown class descriptions are unavailable.

**Use in our experiments:**
If using CLAP or AudioCLIP, adapt UTL by learning unknown audio-text prompts. Known classes are represented by class text prompts, and unknown rejection is performed using learned unknown text representations.

**Strength:** strong few-shot open-set method in the foundation-model setting.
**Limitation:** originally CLIP-based vision method; direct use requires audio-text model adaptation.

---

### 3.3 FOAC-AIFP, TASLP 2026

FOAC-AIFP is the most directly related audio few-shot open-set classification method. It designs an encoder and a classifier consisting of prototype generators for few-shot classes and open-set classes. It recognizes seen query samples and rejects unseen query samples.

**Use in our experiments:**
Use FOAC-AIFP as an audio open-set module. For each session, construct few-shot prototypes and open-set prototypes, then use its open-set score to divide known and unknown samples. For incremental evaluation, newly discovered classes can be appended by the chosen CIL method.

**Strength:** audio-specific and directly related to few-shot open-set audio classification.
**Limitation:** mainly episode-level open-set recognition; does not natively perform online class discovery and continual expansion.

---

### 3.4 OAFN, KBS 2025

OAFN is an audio-specific open-world audio few-shot learning network for event classification. It focuses on label noise and environmental noise in real-world audio datasets. It uses MobileNetV3, transfer learning, dual-channel calibration, inter-class/intra-class calibration, and data perturbation.

**Use in our experiments:**
Use OAFN as a robust audio few-shot feature and similarity baseline. For open-set detection, an unknown threshold can be placed on its final EC similarity or prediction confidence. For online learning, combine the OAFN feature space with a selected incremental classifier.

**Strength:** audio-specific; robust to label noise and additive noise.
**Limitation:** not a complete open-world continual learning method; unknown discovery and classifier expansion need to be added.

---

## 4. Existing Open-World / Open-World Continual Learning Methods

### 4.1 OWCL / MORE, Artificial Intelligence 2025

This work theoretically unifies novelty detection and continual learning. It proposes that strong OOD detection is necessary for successful class-incremental learning and designs methods such as HAT+CSI, Sup+CSI, and MORE. MORE uses replay data as OOD data for current-task OOD learning and performs open-world continual learning.

**Use in our experiments:**
Use OWCL/MORE as a direct open-world continual learning baseline. It can be adapted to audio features by replacing the image backbone with an audio encoder and using session-wise task heads.

**Strength:** directly targets open-world continual learning.
**Limitation:** not audio-specific and assumes task/session training data are available.

---

### 4.2 Buffer-Free CIL with OOD Detection, arXiv 2025

This work studies class-incremental learning with OOD detection in open-world scenarios, especially without a replay buffer. It investigates post-hoc OOD detection methods as substitutes for buffer-based OOD detection.

**Use in our experiments:**
Use it as a privacy-preserving or memory-free baseline. It is useful if we want to show that our prototype memory is more efficient and more suitable for few-shot audio streams than full replay.

**Strength:** directly links CIL and OOD detection.
**Limitation:** not few-shot audio-specific.

---

### 4.3 OpenHAIV, arXiv 2025

OpenHAIV is a practical open-world learning framework integrating OOD detection, new class discovery, and incremental continual fine-tuning. Its pipeline is conceptually close to our detect–discover–adapt process.

**Use in our experiments:**
Use OpenHAIV as an end-to-end open-world baseline if implementation details are available. Otherwise, use it as a conceptual comparison in related work.

**Strength:** complete open-world pipeline.
**Limitation:** not audio-specific; may require substantial adaptation.

---

### 4.4 AFCIL / CrossWorld-CL, arXiv 2025

Annotation-Free Class-Incremental Learning considers unlabeled sequential data and aims to acquire new classes without supervision. CrossWorld-CL uses external world knowledge to guide class discovery and continual learning.

**Use in our experiments:**
This is useful as an unlabeled incremental learning baseline, especially if we want to compare against methods that do not assume labels during sessions.

**Strength:** closer to unlabeled open-world streams.
**Limitation:** relies on external world knowledge and is not audio-specific.

---

### 4.5 Open-World Sound Event Detection, arXiv 2026

This work studies open-world sound event detection, where unknown events can be detected, labeled by a human oracle, and incrementally integrated into the model. It is relevant because it is audio-specific and explicitly discusses open-world sound events.

**Use in our experiments:**
Use it as an audio open-world reference method if code and details are sufficient. It may be better used as related work if the task is sound event detection rather than classification.

**Strength:** audio-specific open-world direction.
**Limitation:** task setting may differ from few-shot open-world audio classification.

---

## 5. Recommended Main 4 × 4 Baseline Matrix

For the main comparison, I recommend using four incremental learning methods and four open-set methods.

### Incremental learning axis

1. PITEL-CUSC / pseudo-incremental stochastic classifier, TASLP 2025
2. Fully-FCAC / multi-level embedding extractor + ridge regression classifier, 2025
3. Tri-WE, CVPR 2025
4. MACIL, ICML/arXiv 2025

### Open-set recognition axis

1. COSTARR, ICCV 2025
2. UTL, ICCV 2025
3. FOAC-AIFP, TASLP 2026
4. OAFN-score, KBS 2025

### 16 combined baselines

| Incremental method | OSR method | Baseline name          |
| ------------------ | ---------- | ---------------------- |
| PITEL-CUSC         | COSTARR    | PITEL-CUSC × COSTARR   |
| PITEL-CUSC         | UTL        | PITEL-CUSC × UTL       |
| PITEL-CUSC         | FOAC-AIFP  | PITEL-CUSC × FOAC-AIFP |
| PITEL-CUSC         | OAFN-score | PITEL-CUSC × OAFN      |
| Fully-FCAC         | COSTARR    | Fully-FCAC × COSTARR   |
| Fully-FCAC         | UTL        | Fully-FCAC × UTL       |
| Fully-FCAC         | FOAC-AIFP  | Fully-FCAC × FOAC-AIFP |
| Fully-FCAC         | OAFN-score | Fully-FCAC × OAFN      |
| Tri-WE             | COSTARR    | Tri-WE × COSTARR       |
| Tri-WE             | UTL        | Tri-WE × UTL           |
| Tri-WE             | FOAC-AIFP  | Tri-WE × FOAC-AIFP     |
| Tri-WE             | OAFN-score | Tri-WE × OAFN          |
| MACIL              | COSTARR    | MACIL × COSTARR        |
| MACIL              | UTL        | MACIL × UTL            |
| MACIL              | FOAC-AIFP  | MACIL × FOAC-AIFP      |
| MACIL              | OAFN-score | MACIL × OAFN           |

If the implementation of UTL is too difficult because it requires CLIP-like text prompts, replace UTL with MLS as a strong and simple open-set scoring baseline. If the implementation of OAFN-score is not accepted as a strict OSR method, keep OAFN as an audio-specific open-world few-shot baseline and use TANE or MLS as the fourth OSR method.

---

## 6. Recommended Direct Open-World Baselines

In addition to the 16 combined baselines, I recommend adding 3–4 direct open-world methods:

1. OWCL / MORE
2. OpenHAIV
3. Buffer-free CIL with OOD Detection
4. Open-World Sound Event Detection

These methods are important because they are not merely combinations of CIL and OSR modules. They directly target open-world or open-world continual learning.

---

## 7. Implementation Protocol for Fair Comparison

To make the comparison fair, use two experimental protocols.

### Protocol A: Shared encoder protocol

All baselines use the same audio encoder. Only the incremental update module and OSR module are replaced. This protocol tests whether our classifier and open-world learning strategy are better under the same representation.

Recommended encoder choices:

1. ResNet audio spectrogram encoder
2. AST encoder
3. Your uncertainty-guided encoder

This protocol is the fairest for method comparison.

### Protocol B: Native method protocol

Each baseline uses its original feature extractor and classifier design. This protocol tests the full method performance, but it is less controlled because the backbone, pretraining, and optimization differ.

---

## 8. Session-wise Evaluation Procedure

For each online session:

1. Extract embeddings for incoming audio samples.
2. Use the OSR method to split samples into known and unknown pools.
3. Evaluate known-class recognition accuracy.
4. Cluster unknown samples into new classes.
5. Use a small number of samples or pseudo-labels from each cluster to update the incremental classifier.
6. Expand the class space and repeat in the next session.

Recommended metrics:

1. S0 base accuracy
2. Session-wise incremental accuracy: S1_inc, S2_inc, S3_inc, S4_inc
3. Session-wise all-class accuracy: S1_all, S2_all, S3_all, S4_all
4. Average incremental accuracy: AA_inc
5. Average all-class accuracy: AA_all
6. Performance dropping rate: PD_inc, PD_all
7. Open-set metrics: AUROC, OSCR, FPR95
8. Unknown discovery metrics: clustering ACC, NMI, ARI

---

## 9. Recommended Final Baseline Set

For the main paper, I recommend the following baseline groups:

### Group 1: Classic sanity baselines

1. ProtoNet + MLS
2. CEC + MLS
3. CEC + TANE
4. PAN + MLS

These are not all 2025–2026 methods, but they provide interpretable lower and middle baselines.

### Group 2: Strong 2025–2026 combined baselines

1. PITEL-CUSC × COSTARR
2. PITEL-CUSC × FOAC-AIFP
3. Fully-FCAC × COSTARR
4. Fully-FCAC × FOAC-AIFP
5. Tri-WE × COSTARR
6. Tri-WE × UTL
7. MACIL × COSTARR
8. MACIL × FOAC-AIFP

If page space allows, report the full 4 × 4 matrix in the supplementary material.

### Group 3: Direct open-world baselines

1. OWCL / MORE
2. OpenHAIV
3. Buffer-free CIL with OOD Detection
4. OAFN or Open-World SED as audio-specific open-world reference

---

## 10. Summary

The most suitable baseline design is not to compare only with ordinary few-shot or class-incremental methods. Instead, the comparison should reflect the open-world nature of the task. Therefore, the strongest experimental setup should include:

1. Audio-specific FCAC methods as incremental baselines.
2. Recent OSR and FSOR methods as unknown rejection baselines.
3. 2025–2026 open-world continual learning methods as direct open-world baselines.
4. A 4 × 4 combination matrix showing whether simply combining strong incremental learning and strong open-set recognition is sufficient.

The expected conclusion is that existing incremental methods can update the classifier but lack unknown detection, while existing open-set methods can reject unknown samples but do not support online discovery and classifier evolution. Direct open-world methods are closer to our setting, but they are mostly vision-based or not few-shot audio-specific. Our method is different because it jointly learns a robust encoder, class-wise dual-prototype open-set classifier, CFFM-enhanced unknown discovery, and online prototype gallery expansion for few-shot open-world audio classification.
