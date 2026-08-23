# LS-100 training-free DFSB open-set result

Protocol: frozen warm-up encoder and frozen K=128 DFSB; no learned open-set head.
Class semantic/structural prototypes come from the 40,000-sample base train split.
Score normalization and 95% rejection thresholds use only 12,000 base-validation
samples. The 8,000 known and 2,000 novel test samples are used only for evaluation.

| Score | AUROC | AUPR-U | FPR95 | Known acceptance | Unknown recall | Balanced OSR acc. |
|---|---:|---:|---:|---:|---:|---:|
| Semantic only | 0.8636 | 0.5586 | 0.4569 | 0.9659 | 0.2180 | 0.5919 |
| Semantic + structural | 0.8607 | 0.5431 | 0.4564 | 0.9669 | 0.2030 | 0.5849 |
| Semantic + structural + residual | 0.8623 | 0.5506 | 0.4709 | 0.9660 | 0.2215 | 0.5938 |

Fixed weights were semantic/structural/residual = 0.7/0.2/0.1. They were not tuned on
novel test labels.

Conclusion: the frozen bank does not improve ranking-based OSR metrics before
structure-guided representation training. Its response is likely redundant with the
semantic embedding. This result motivates, rather than replaces, Phase C: masked
structure prediction must make shared feature structure influence representation
learning. The training-free variants remain required ablations.
