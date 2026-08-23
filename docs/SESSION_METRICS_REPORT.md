# Current session-wise metrics

All values are percentages. Session 0 is the base-class evaluation; its incremental/open-set metrics are not defined and are shown as 0.00. LS-100 and NS-100 are the reproducible 50-repeat audits. FSC-89 is the current non-oracle support-prototype 50-repeat audit (`novel_bias=-0.05`, reflow=0.10).

## LS-100

| metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA |
|---|---:|---:|---:|---:|---:|---:|
| Inc_acc | 0.00 | 92.38 | 78.21 | 62.31 | 60.18 | 73.27 |
| All_acc | 93.67 | 93.29 | 91.53 | 88.07 | 86.28 | 89.79 |
| AUROC | — | 99.96 | 98.43 | 83.50 | 95.70 | 94.40 |
| F1-score | — | 98.25 | 94.45 | 81.40 | 87.52 | 90.40 |

## NS-100

| metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA |
|---|---:|---:|---:|---:|---:|---:|
| Inc_acc | 0.00 | 66.90 | 70.78 | 74.23 | 68.72 | 70.16 |
| All_acc | 99.85 | 97.87 | 96.58 | 95.76 | 93.57 | 95.94 |
| AUROC | — | 97.80 | 91.32 | 95.12 | 92.27 | 94.12 |
| F1-score | — | 80.52 | 76.66 | 79.07 | 84.80 | 80.26 |

## FSC-89

| metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA |
|---|---:|---:|---:|---:|---:|---:|
| Inc_acc | 0.00 | 41.80 | 44.10 | 38.80 | 35.88 | 40.14 |
| All_acc | 55.54 | 50.22 | 47.09 | 44.69 | 42.94 | 46.24 |
| AUROC | — | 97.07 | 89.15 | 92.31 | 86.72 | 91.31 |
| F1-score | — | 86.19 | 65.58 | 77.33 | 70.77 | 74.97 |

Sources: `logs/ls100_rank25_late_frozen_50.log`, `logs/ns_all_q50_cana_50final.log`, and `docs/FSC89_50_REPEAT_AUDIT.md`.
