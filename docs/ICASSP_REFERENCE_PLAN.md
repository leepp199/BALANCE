# ICASSP reference plan (relative to the original FOWAC draft)

The original DOCX already cites audio classification, few-shot open-set keyword
spotting, FSCIL audio methods, OpenMax and TANE. Those references should be retained,
but the revised paper needs a tighter set of citations aligned with the new claims.

## Must cite in the revised main text

| Topic | Reference | Where it supports our paper |
|---|---|---|
| Few-shot prototype calibration | Wang et al., TEEN (NeurIPS 2023) | training-free novel prototype calibration baseline and motivation |
| Continual category discovery | Wu et al., MetaGCD (ICCVW 2023) | mixed known/novel continual discovery formulation |
| Online continual discovery | Park et al., OCGCD (ECCV 2024) | distinction between offline and streaming discovery |
| Debiased category discovery | Ma et al., Happy (NeurIPS 2024) | recent SOTA transferred baseline |
| Projected distillation | Rype\v{s}c et al., CaMP (ECCV 2024) | recent continual discovery baseline / related work |
| Bayesian covariance-aware discovery | Dai and Chauhan, VB-CGCD (ICML 2025) | recent SOTA baseline and covariance-misalignment motivation |
| Open-set recognition | Bendale and Boult, OpenMax (CVPR 2016) | classical EVT-based rejection baseline |
| Open-set incremental learning | Xu et al., OpenIncrement (ICCVW 2023) | closest unified open-set + class-incremental framing |
| MC Dropout uncertainty | Gal and Ghahramani (ICML 2016) | uncertainty estimator implementation |
| Nuclear norm | Deng et al. (ICML 2023) | nuclear norm as confidence/dispersity diagnostic, not raw uncertainty |
| Replay/proximal retention | Yoo et al., Layerwise Proximal Replay (ICML 2024) | anti-forgetting rationale and replay comparison |
| Audio FSCIL | Li et al. stochastic classifier; Xie et al. discriminative prototypes; Li et al. AMFO | audio-domain baselines and forgetting/overfitting discussion |

## Cite only if used in the final experiment table

YLOC/PAL, OPCR and any 2026 methods should be cited only if the final manuscript
retains their transferred baseline rows. Their protocols are visual or generic CGCD
protocols, so the text must state that our acoustic results are component transfers,
not direct reproductions.

## Claims that must be rewritten

* State the nuanced version: the uncentered nuclear norm characterizes confidence and
  prediction dispersity, while the nuclear norm of the centered MC prediction matrix
  is used as a disagreement-based uncertainty component alongside entropy and mutual
  information.
* Do not claim meta-training improves open-set generalization until held-out AUROC
  selection beats the no-meta row.
* Do not claim replay prevents forgetting from AA-inc alone; report old-class
  retention and forgetting curves.

All entries listed above have been added to `paper/icassp2027/references.bib`.
