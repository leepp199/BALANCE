# Current three-dataset session report (%)

AA is the arithmetic mean of Sessions 1--4. Session0 is reported separately and
is not included in AA. `N/A` means the metric is undefined before incremental
sessions.

## LS-100 — frozen 50-repeat result

| Metric | Session0 | Session 1 | Session 2 | Session 3 | Session 4 | AA |
|---|---:|---:|---:|---:|---:|---:|
| Inc_acc | N/A | 92.38 | 78.21 | 62.31 | 60.18 | 73.27 |
| All_acc | 93.67 | 93.29 | 91.53 | 88.07 | 86.28 | 89.79 |
| AUROC | N/A | 99.96 | 98.43 | 83.50 | 95.70 | 94.40 |
| F1-score | N/A | 98.25 | 94.45 | 81.40 | 87.52 | 90.40 |
| Known accuracy | N/A | 91.64 | 94.09 | 75.72 | 92.57 | 88.50 |
| Unknown accuracy | N/A | 96.28 | 79.51 | 60.24 | 72.80 | 77.21 |

Source: `logs/ls100_rank25_late_frozen_50.log`.

## NS-100 — frozen 50-repeat result

| Metric | Session0 | Session 1 | Session 2 | Session 3 | Session 4 | AA |
|---|---:|---:|---:|---:|---:|---:|
| Inc_acc | N/A | 66.90 | 70.78 | 74.23 | 68.72 | 70.16 |
| All_acc | 99.85 | 97.87 | 96.58 | 95.76 | 93.57 | 95.94 |
| AUROC | N/A | 97.80 | 91.32 | 95.12 | 92.27 | 94.12 |
| F1-score | N/A | 80.52 | 76.66 | 79.07 | 84.80 | 80.26 |
| Known accuracy | N/A | 76.77 | 72.28 | 76.50 | 83.27 | 77.21 |
| Unknown accuracy | N/A | 86.48 | 89.20 | 92.96 | 78.64 | 86.82 |

Source: `logs/ns_all_q50_cana_50final.log`.

## FSC-89 — frozen 50-repeat non-oracle result

| Metric | Session0 | Session 1 | Session 2 | Session 3 | Session 4 | AA |
|---|---:|---:|---:|---:|---:|---:|
| Inc_acc | N/A | 40.00 | 42.15 | 37.53 | 34.63 | 38.58 |
| All_acc | 55.54 | 50.65 | 47.77 | 45.52 | 43.74 | 46.92 |
| AUROC | N/A | 97.36 | 90.20 | 91.91 | 87.87 | 91.84 |
| F1-score | N/A | 86.39 | 65.67 | 77.57 | 71.66 | 75.32 |
| Known accuracy | N/A | 81.42 | 64.63 | 70.92 | 66.84 | 70.95 |
| Unknown/discovery accuracy | N/A | 49.94 | 58.32 | 49.03 | 40.45 | 49.44 |

Source: 50 complete repeats selected from the main log (first 30 repeats),
`..._reflow0.2_p1.log` (10 repeats), and `..._reflow0.2_p2.log` (10 repeats).
All runs use `oracle_cluster=False`, offline local checkpoints, and labeled
few-shot support prototypes generated only by the current model's `model.encode`.
The result is stable and satisfies `All_acc > Inc_acc`, but aggregate
`46.92 / 38.58` remains slightly below the requested approximate `48 / 40`
target, so the broader task remains active.
