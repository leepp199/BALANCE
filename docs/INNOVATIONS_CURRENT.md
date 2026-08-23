# Current paper-facing innovations and naming

## Problem setting

FOWAC targets open-world few-shot class-incremental audio classification. Each
session mixes registered classes and emerging classes. The learner must reject
unknown inputs, discover emerging categories, register them from few-shot
support, and preserve earlier classes.

## Final module names

| Paper name | Meaning | Implementation aliases |
|---|---|---|
| **LSRB — Latent Structure Reference Bank** | Frozen cross-sample references learned during base training in the task-adapted latent space | `DFSB` |
| **CANA — Capacity-Aware Novel-Class Assignment** | Label-free balanced assignment of rejected candidates to emerging-class slots | `balanced_kmeans`, `BCD` |
| **Support Prototype Bank** | Five-shot embeddings produced by the current model's `model.encode`, retained per novel class | `_LABELED_SUPPORT_PROTOS`, `_NOVEL_SUPPORT_BANK` |
| **Margin-based Open-World Gate** | Adaptive known/unknown decision before novel-class routing | `threshold_free.py`, OSR margin logic |
| **Exemplar-Free Uncertainty Memory** | Per-class moments and reliability used for guarded statistical replay | `stat_memory` |

## Contributions

1. **Unified open-world continual protocol.** FOWAC combines rejection, novel-class
   discovery, few-shot registration, dynamic classifier expansion, and old-class
   retention in one repeated mixed-stream protocol.
2. **LSRB representation reference.** A cross-sample latent structure bank is learned
   once during base training and reused as a stable reference across sessions. It is
   not interpreted as a physical time/frequency bank.
3. **CANA discovery and registration.** Rejected candidates are assigned to balanced
   emerging-class slots without evaluation labels. Class identity is aligned from the
   current few-shot support episode.
4. **Current-encoder support prototypes.** Final novel prototypes use only the current
   `model.encode` embeddings of five-shot support samples. The complete support bank is
   retained for top-k or temperature-weighted scoring; discovered clusters are not
   silently substituted for it.
5. **Margin-based open-set routing.** A label-free adaptive margin separates registered
   samples from rejected candidates. Rejected candidates are then routed to the novel
   support bank rather than compared with a single artificial unknown prototype.
6. **Reliability-gated exemplar-free retention.** Class moments, confidence anchors and
   bounded residual updates protect old classes while avoiding harmful replay when
   representation statistics are unreliable.
7. **Traceable evaluation.** Session-0 accuracy, Inc/All accuracy, AUROC, F1, AUPR,
   FPR95 and clustering diagnostics are recorded under an offline, non-oracle protocol.

## Naming rules for figures and manuscript

Use LSRB and CANA in all paper-facing figures, captions and tables. Do not label a
line or module as `TEEN` unless it is an actual baseline. Do not draw a separate
unknown prototype in the current method overview: the current implementation uses
adaptive margin rejection followed by the novel support bank.
