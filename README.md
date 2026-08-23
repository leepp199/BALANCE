# FOWAC open-world continual audio classification

This repository contains the FOWAC implementation and the reproducibility
materials for the ICASSP study. The main training/evaluation entry point is
`train_unopenset.py`. Dataset-specific configurations are under `configs/`;
reproducible experiment wrappers are under `scripts/`.

## Offline execution

The experiments are designed to run without network access. Before running an
experiment, execute:

```bash
bash scripts/offline_preflight.sh
```

The FSC-89 non-oracle candidate wrapper is:

```bash
bash scripts/run_fsc89_candidate.sh
```

It uses the local checkpoint and current-model `model.encode` support
embeddings. Evaluation labels are used only for reporting metrics.

## Current evidence

The current corrected FSC-89 audit and session-wise tables are documented in
`docs/FSC89_50_REPEAT_AUDIT.md` and `docs/SESSION_METRICS_REPORT.md`. The
corrected 10-repeat audit is recorded in
`logs/fsc89_candidate_corrected_t02_r10.log` when experiment outputs are kept
locally; generated logs are intentionally excluded from version control.

## Main components

- dynamic open-world classifier expansion;
- support-aligned novel-class discovery without oracle labels;
- current-encoder few-shot prototype banks;
- margin-based known/unknown rejection;
- open-set and continual-learning metrics (AUROC, F1, Inc/All accuracy).

See `docs/OFFLINE_REPRODUCIBILITY.md` and `docs/PAPER_RESULT_TABLES_CURRENT.md`
for protocol details and result provenance.
