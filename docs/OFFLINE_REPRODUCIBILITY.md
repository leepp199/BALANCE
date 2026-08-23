# Offline reproducibility contract

All ICASSP screening and formal runs must execute without network access.

- `MYNET` never downloads ImageNet weights during construction. A complete local
  experiment checkpoint restores the encoder and classifier state.
- Run `bash scripts/offline_preflight.sh` before every experiment queue.
- Dataset roots, checkpoints, configurations, seeds, commands, raw session records,
  dependency versions, and file hashes must be archived locally.
- Missing assets are fatal. A runner must not silently download a model or dataset,
  fall back to random initialization, or replace missing samples.
- Screening seeds and formal 50-repeat test streams are separate. Hyperparameters are
  frozen before the formal run.

The required environment flags are `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=offline`, and `WANDB_DISABLED=true`.
