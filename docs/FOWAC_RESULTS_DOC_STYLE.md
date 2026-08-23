# FOWAC results in the manuscript table format

> The session/AA tables follow the supplied document. FSC-89 now uses the repaired support-aligned non-oracle evaluator; prototypes come from the current model.encode and the 5-shot support only.

## Table: FOWAC on LS-100 (%)

| Method | Metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA* |
|---|---|---:|---:|---:|---:|---:|---:|
| FOWAC (ours) | Inc_acc | N/A | 92.38 | 78.21 | 62.31 | 60.18 | 73.27 |
| FOWAC (ours) | All_acc | 93.67 | 93.29 | 91.53 | 88.07 | 86.28 | 89.79 |

## Table: FOWAC on NS-100 (%)

| Method | Metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA* |
|---|---|---:|---:|---:|---:|---:|---:|
| FOWAC (ours) | Inc_acc | N/A | 66.90 | 70.78 | 74.23 | 68.72 | 70.16 |
| FOWAC (ours) | All_acc | 99.85 | 97.87 | 96.58 | 95.76 | 93.57 | 95.94 |

## Table: FOWAC on FSC-89 (%)

| FOWAC (ours) | Metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA |
|---|---|---:|---:|---:|---:|---:|---:|
| FOWAC (ours) | Inc_acc | N/A | 41.80 | 44.10 | 38.80 | 35.88 | 40.14 |
| FOWAC (ours) | All_acc | 55.54 | 50.22 | 47.09 | 44.69 | 42.94 | 46.24 |
| FOWAC (ours) | AUROC | N/A | 97.07 | 89.15 | 92.31 | 86.72 | 91.31 |
| FOWAC (ours) | F1-score | N/A | 86.19 | 65.58 | 77.33 | 70.77 | 74.97 |

## Additional open-set table (%)

| Dataset | Metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA* |
|---|---|---:|---:|---:|---:|---:|---:|
| LS-100 | AUROC | N/A | 99.96 | 98.43 | 83.50 | 95.70 | 94.40 |
| LS-100 | F1-score | N/A | 98.25 | 94.45 | 81.40 | 87.52 | 90.40 |
| LS-100 | Known accuracy | N/A | 91.64 | 94.09 | 75.72 | 92.57 | 88.50 |
| LS-100 | Unknown accuracy | N/A | 96.28 | 79.51 | 60.24 | 72.80 | 77.21 |
| NS-100 | AUROC | N/A | 97.80 | 91.32 | 95.12 | 92.27 | 94.12 |
| NS-100 | F1-score | N/A | 80.52 | 76.66 | 79.07 | 84.80 | 80.26 |
| NS-100 | Known accuracy | N/A | 76.77 | 72.28 | 76.50 | 83.27 | 77.21 |
| NS-100 | Unknown accuracy | N/A | 86.48 | 89.20 | 92.96 | 78.64 | 86.82 |

*AA is the mean of incremental Sessions 1–4, matching the supplied manuscript. Session 0 is the base-session all-class accuracy; N/A metrics are undefined before open-world increments.

The promoted balanced FSC-89 audit uses `novel_bias=-0.05` and 50 complete
offline repeats: `Inc_acc=40.14%`, `All_acc=46.24%`, AUROC `91.31%`, and F1
`74.97%`. Full provenance is in `FSC89_50_REPEAT_AUDIT.md`.
