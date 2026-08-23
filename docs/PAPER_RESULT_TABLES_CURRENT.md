# Current paper result tables

> Generated from traceable local logs. FSC-89's previously promoted 50-repeat row is retained provisionally for provenance; a support-bank overwrite bug was fixed afterward. The corrected 10-repeat audit is now the active diagnostic, while a corrected 50-repeat audit is still required for the final paper row.

## Protocol-compatible main comparison (%)

| method | dataset | provenance                          | repeats | inc_aa | all_aa | auroc_aa | osr_f1_aa |
| ------ | ------- | ----------------------------------- | ------- | ------ | ------ | -------- | --------- |
| Happy  | FSC-89  | official-component-reimplementation | 10      | 5.56   | 16.17  | 39.14    | 14.29     |
| OFCL   | FSC-89  | official-component-transfer         | 10      | 0.73   | 45.95  | 33.34    | 0.00      |
| OPCR   | FSC-89  | paper-reimplementation              | 10      | 1.02   | 46.58  | 66.82    | 66.89     |
| TEEN   | FSC-89  | official-formula-transfer           | 10      | 23.05  | 37.44  | 67.10    | 63.01     |
| YLOC   | FSC-89  | paper-reimplementation              | 10      | 0.00   | 48.56  | 30.24    | 51.58     |
| FOWAC  | FSC-89  | support-aligned-model-encode-50     | 50      | 40.14  | 46.24  | 91.31    | 74.97     |
| FOWAC  | LS-100  | proposed-formal-50                  | 50      | 73.27  | 89.79  | 94.40    | 90.40     |
| Happy  | LS-100  | official-component-reimplementation | 10      | 37.46  | 58.06  | 65.42    | 31.78     |
| OFCL   | LS-100  | official-component-transfer         | 10      | 50.81  | 79.73  | 89.33    | 79.25     |
| OPCR   | LS-100  | paper-reimplementation              | 10      | 43.27  | 79.61  | 85.61    | 80.41     |
| TEEN   | LS-100  | official-formula-transfer           | 10      | 69.67  | 89.41  | 90.31    | 84.69     |
| YLOC   | LS-100  | paper-reimplementation              | 10      | 52.17  | 80.01  | 79.97    | 79.25     |
| FOWAC  | NS-100  | proposed-formal-50                  | 50      | 70.16  | 95.94  | 94.12    | 80.26     |
| Happy  | NS-100  | official-component-reimplementation | 10      | 28.34  | 63.71  | 56.52    | 23.96     |
| OFCL   | NS-100  | official-component-transfer         | 10      | 59.86  | 94.50  | 93.78    | 89.89     |
| OPCR   | NS-100  | paper-reimplementation              | 10      | 43.13  | 92.75  | 95.54    | 91.71     |
| TEEN   | NS-100  | official-formula-transfer           | 10      | 52.45  | 93.58  | 89.42    | 77.37     |
| YLOC   | NS-100  | paper-reimplementation              | 10      | 60.17  | 94.71  | 95.26    | 89.37     |

## Open-set recognition and continual metrics (%)

| method | dataset | repeats | known | unknown | auroc | aupr  | fpr95 | f1    | inc   | all   | source                               |
| ------ | ------- | ------- | ----- | ------- | ----- | ----- | ----- | ----- | ----- | ----- | ------------------------------------ |
| TEEN   | FSC-89  | 10      | 69.91 | 32.43   | 67.10 | 67.44 | 52.00 | 63.01 | 23.05 | 37.44 | logs/raw_v2_teen_fsc89_10runs.log    |
| FOWAC  | LS-100  | 50      | 88.50 | 77.21   | 94.40 | 95.53 | 26.48 | 90.40 | 73.27 | 89.79 | logs/ls100_rank25_late_frozen_50.log |
| TEEN   | LS-100  | 10      | 82.34 | 67.86   | 90.31 | 91.47 | 34.50 | 84.69 | 69.67 | 89.41 | logs/raw_v2_teen_ls100_10runs.log    |
| FOWAC  | NS-100  | 50      | 77.21 | 86.82   | 94.12 | 94.28 | 23.10 | 80.26 | 70.16 | 95.94 | logs/ns_all_q50_cana_50final.log     |
| TEEN   | NS-100  | 10      | 75.02 | 72.70   | 89.42 | 90.84 | 36.40 | 77.37 | 52.45 | 93.58 | logs/raw_v2_teen_ns100_10runs.log    |

## Visual-domain methods retained as protocol-only references

| method        | provenance                            | status                          |
| ------------- | ------------------------------------- | ------------------------------- |
| MetaGCD       | original visual C-GCD protocol        | not numerically mixed           |
| OCGCD/DEAN    | original online visual C-GCD protocol | not numerically mixed           |
| OpenIncrement | original visual OSR+CIL protocol      | not numerically mixed           |
| CAMP          | ECCV 2024 visual GCCD protocol        | not numerically mixed           |
| VB-CGCD       | audio transfer attempted              | DNF: singular 5-shot covariance |
| FaE           | AAAI 2026 visual C-GCD protocol       | not numerically mixed           |
| PRISM         | ICLR 2026 visual OW-CCD protocol      | not numerically mixed           |
| OCCD          | CVPR 2026 visual-drift protocol       | not numerically mixed           |
| VC-CGCD       | 2026 visual C-GCD preprint protocol   | not numerically mixed           |
