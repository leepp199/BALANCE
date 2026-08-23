# Generated experiment tables

> Generated from CSV ledgers; audit and screening rows are excluded from main tables.
>
> **Historical tables below are retained for provenance.** The current repaired FSC-89 audit is recorded in `PAPER_RESULT_TABLES_CURRENT.md` and `FSC89_50_REPEAT_AUDIT.md`; do not use the withdrawn rows below as the current result.
> and every paired-significance value in this generated snapshot came from a legacy
> cumulative-mean aggregation bug. They must not be cited. Happy/OFCL/OPCR/YLOC rows
> come from independent per-seed JSON files and remain valid. This file will be
> regenerated only from complete `[RAW]` round/session records.

## Main continual-learning comparison

| Method | Dataset | AA-inc (%) | Final-inc (%) | AA-all (%) | Repeat unit |
|---|---:|---:|---:|---:|---|
| TEEN | FSC-89 | 24.23 | 16.33 | 37.52 | data-stream repeats |
| TEEN | LS-100 | 69.26 | 55.96 | 89.39 | data-stream repeats |
| TEEN | NS-100 | 51.45 | 49.60 | 93.44 | data-stream repeats |
| Happy-HAProto-acoustic | LS-100 | 37.46 ± 2.03 | 23.66 ± 3.55 | 58.06 ± 0.50 | algorithm seeds (n=10) |
| Happy-HAProto-acoustic | NS-100 | 28.34 ± 1.84 | 17.57 ± 3.21 | 63.71 ± 0.29 | algorithm seeds (n=10) |
| Happy-HAProto-acoustic | FSC-89 | 5.56 ± 0.21 | 5.15 ± 0.40 | 16.17 ± 0.11 | algorithm seeds (n=10) |
| OFCL-acoustic-transfer | LS-100 | 50.81 ± 2.95 | 40.42 ± 2.67 | 79.73 ± 0.47 | algorithm seeds (n=10) |
| OFCL-acoustic-transfer | NS-100 | 59.86 ± 1.24 | 59.83 ± 1.99 | 94.50 ± 0.18 | algorithm seeds (n=10) |
| OFCL-acoustic-transfer | FSC-89 | 0.73 ± 0.44 | 1.02 ± 0.41 | 45.95 ± 0.09 | algorithm seeds (n=10) |
| OPCR-acoustic | LS-100 | 43.27 ± 1.37 | 32.00 ± 1.57 | 79.61 ± 0.21 | algorithm seeds (n=10) |
| OPCR-acoustic | NS-100 | 43.13 ± 0.90 | 49.57 ± 3.72 | 92.75 ± 0.18 | algorithm seeds (n=10) |
| OPCR-acoustic | FSC-89 | 1.02 ± 0.31 | 0.75 ± 0.13 | 46.58 ± 0.04 | algorithm seeds (n=10) |
| YLOC-PAL-acoustic | LS-100 | 52.17 ± 2.15 | 38.91 ± 1.82 | 80.01 ± 0.31 | algorithm seeds (n=10) |
| YLOC-PAL-acoustic | NS-100 | 60.17 ± 2.30 | 62.29 ± 0.83 | 94.71 ± 0.22 | algorithm seeds (n=10) |
| YLOC-PAL-acoustic | FSC-89 | 0.00 ± 0.01 | 0.01 ± 0.02 | 48.56 ± 0.00 | algorithm seeds (n=10) |
| FOWAC-UMR | FSC-89 | 24.23 | 16.33 ± 0.64 | 37.52 | paired data-stream repeats |
| FOWAC-UMR | NS-100 | 51.45 | 49.60 ± 1.12 | 93.44 | paired data-stream repeats |
| FOWAC-UMR | LS-100 | 70.62 | 56.69 ± 2.55 | 89.64 | paired data-stream repeats |
| FOWAC-DS | LS-100 | 71.78 | 60.82 ± 1.61 | 89.76 | paired data-stream repeats |
| FOWAC-BCD | FSC-89 | 26.38 | 17.09 ± 0.48 | 37.81 | paired data-stream repeats |

## Paired significance versus TEEN

| Method | Dataset | Metric | Gain (pp) | 95% CI (pp) | paired-t p | Wilcoxon p |
|---|---|---|---:|---:|---:|---:|
| FOWAC-UMR | LS-100 | inc | 1.36 | [0.86, 1.86] | 0.0007367 | 0.001953 |
| FOWAC-UMR | LS-100 | final_inc | 0.72 | [0.14, 1.27] | 0.04445 | 0.04883 |
| FOWAC-UMR | LS-100 | all | 0.24 | [0.17, 0.32] | 0.0002297 | 0.001953 |
| FOWAC-UMR | NS-100 | inc | 0.00 | [0.00, 0.00] | nan | nan |
| FOWAC-UMR | NS-100 | final_inc | 0.00 | [0.00, 0.00] | nan | nan |
| FOWAC-UMR | NS-100 | all | 0.00 | [0.00, 0.00] | nan | nan |
| FOWAC-UMR | FSC-89 | inc | 0.00 | [0.00, 0.00] | nan | nan |
| FOWAC-UMR | FSC-89 | final_inc | 0.00 | [0.00, 0.00] | nan | nan |
| FOWAC-UMR | FSC-89 | all | 0.00 | [0.00, 0.00] | nan | nan |
| FOWAC-DS | LS-100 | inc | 2.52 | [1.73, 3.15] | 0.0001166 | 0.003906 |
| FOWAC-DS | LS-100 | final_inc | 4.86 | [3.78, 5.64] | 6.339e-06 | 0.001953 |
| FOWAC-DS | LS-100 | all | 0.37 | [0.27, 0.45] | 2.845e-05 | 0.001953 |
| FOWAC-BCD | FSC-89 | inc | 2.15 | [1.62, 2.70] | 4.56e-05 | 0.001953 |
| FOWAC-BCD | FSC-89 | final_inc | 0.76 | [0.30, 1.20] | 0.0128 | 0.01367 |
| FOWAC-BCD | FSC-89 | all | 0.28 | [0.23, 0.34] | 5.28e-06 | 0.001953 |

## Metric-complete open-set and discovery evaluation

| Method | Dataset | AUPR (%) | FPR95 (%) ↓ | Clu-ACC (%) | NMI (%) | ARI (%) |
|---|---|---:|---:|---:|---:|---:|
| TEEN | LS-100 | 91.47 ± 3.27 | 34.50 ± 6.64 | 67.86 ± 6.63 | 87.48 ± 1.34 | 74.25 ± 1.65 |
| TEEN | NS-100 | 90.84 ± 5.33 | 36.40 ± 16.43 | 72.70 ± 7.29 | 86.08 ± 2.70 | 71.43 ± 6.07 |
| TEEN | FSC-89 | 67.44 ± 6.44 | 52.00 ± 8.27 | 32.43 ± 5.65 | 66.93 ± 8.47 | 41.08 ± 10.73 |
| FOWAC-BCD | FSC-89 | 72.13 ± 5.00 | 50.10 ± 6.26 | 34.17 ± 6.83 | 59.04 ± 7.11 | 34.53 ± 6.50 |

## Paired extended-metric significance: FOWAC-BCD minus TEEN

| Metric | Gain (pp) | 95% CI (pp) | paired-t p | Wilcoxon p |
|---|---:|---:|---:|---:|
| auroc | 5.05 | [1.17, 8.52] | 0.03344 | 0.06445 |
| aupr | 4.69 | [0.99, 8.16] | 0.03911 | 0.04883 |
| fpr95 | -1.90 | [-4.60, 0.50] | 0.1988 | 0.2053 |
| cluster_acc | 1.74 | [-2.59, 5.72] | 0.4575 | 0.625 |
| nmi | -7.89 | [-13.98, -1.46] | 0.04171 | 0.06445 |
| ari | -6.55 | [-11.77, -0.71] | 0.05517 | 0.06445 |
