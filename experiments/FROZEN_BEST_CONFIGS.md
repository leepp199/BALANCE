# Frozen best configurations (2026-08-22)

All commands set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, use local
checkpoints, and reset classifier state before each independent repeat.

## FSC-89

- Encoder: `/data/lqq/baseline/save/exp_fsc89_69supcon/base_train_for_meta_FMC.pth`
  (SHA-256 `26430943f6b2c503ee0e5df769693f231f0d1a9f6094ef2e1d7d0c0e3feff171`).
- Geometry SHA-256: `3170cb504a8f91aed653530e23d9f1fde9fd8f029f7c4cdd25f07078d0ac16b5`.
- Base split: classes 0-58 use the official validation CSV; zero-based labels
  59-68 reserve a deterministic 400/100 train/validation split. Test is sealed.
- Positive components: moderate base-only SupCon (weight 0.1), final
  `model.encode`, layer4 joint cosine CANA, top1/top2 margin weight 0.5,
  direct reuse of the selected joint assignments, cosine incremental classifier.
- Historical screening result before session-restricted alignment: AA-inc 0.5765
  (`logs/fsc_supcon_cana.log`). It was independently reproduced exactly by
  `scripts/run_frozen_fsc89_5765_audit.sh`, producing 0.6790/0.6290/0.5303/0.4675
  and AA-inc 0.5765 in `logs/fsc89_5765_historical_audit.log`. This is retained
  as an audit result only because it uses the historical unrestricted alignment.
- Corrected session-restricted result under the historical implicit KMeans RNG:
  AA-inc 0.5695 (`logs/fsc_alignfix6.log`).
- Current formally frozen, explicit-seed replay: 0.6180/0.6010/0.5207/0.4375,
  AA-inc 0.5443 (`logs/fsc89_frozen_corrected.log`). The corrected evaluator
  never overwrites a prior session's prototype. This explicit-seed result is the
  scientifically valid and exactly reproducible FSC baseline; 0.5695 remains a
  screening audit rather than a frozen claim.
- Oracle with the same `model.encode`: AA-inc 0.6631
  (`logs/fsc_supcon_oracle.log`).

Rejected and disabled: feature centering, PAN copied verbatim, LDA clustering,
layer2/layer3 replacement, deterministic time-shift TTA, strong SupCon,
agglomerative clustering, TEEN calibration, farthest-shot trimming, and the
robust adapter screen. None is present in the frozen command.

## NS-100

- Formal 50-repeat result: AA-inc 0.701583 (95% CI [0.689890, 0.713277]).
- Source log: `logs/ns_all_q50_cana_50final.log`.
- Checkpoint SHA-256: `13113aa09d8a6c674885f76344757548f1b3246191a4451779c1d214079b5687`.
- Summary: `experiments/ns100_cana_50repeat_summary.json`.
- Positive components: all-candidate discovery, lowest-base-similarity 50%
  filter, capacity-constrained balanced CANA, five refinement iterations.
- The result exceeds the required 0.65 threshold and is frozen; FSC-specific
  encoder adaptation is not migrated because it is unnecessary and risky.

## LS-100

- Current frozen formal 50-repeat result: AA-inc 0.732701, sample standard
  deviation 0.052096; AA-all 0.897931, sample standard deviation 0.007303.
- Session inc_acc means: 0.9238, 0.7821, 0.6231, 0.6018.
- Formal log: `logs/ls100_rank25_late_frozen_50.log`; frozen command:
  `scripts/run_frozen_ls100_rank25_late_50.sh`.
- Highest fixed-stream UMR residual screen: AA-inc 0.7673, final-inc 0.6495
  (`logs/proposed_statmem_residual_ls_screen.log`).
- Existing formal 10-repeat UMR result: AA-inc 0.7089
  (`logs/proposed_statmem_ls100_10runs.log`).
- Existing formal 10-repeat structure-discovery result: AA-inc 0.7139
  (`logs/raw_v2_dfsb_ds_ls100_10runs.log`).
- Positive components retained for the 50-repeat command: current
  `model.encode` for discovery features and prototypes, frozen LSRB structural
  reference, mixed open-world stream, structure weight 0.5, normalized
  clustering, 30 compactness steps, and classifier reset per repeat. Reliability
  routing disables CANA/UMR on LS-100 because both reduced the screening mean.
  Sessions 3–4 use a label-free lowest-knownness top-25 candidate pool from the
  same `model.encode`, preventing threshold drift from changing candidate count.
- LSRB checkpoint SHA-256: `8625bb73c1770496c97bf32d0ff1a783be409bf102ee741c039c3c9341a04628`.
