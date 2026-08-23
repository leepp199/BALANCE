#!/usr/bin/env bash
set -euo pipefail
cd /data/lqq/baseline_dfsb

for seed in $(seq 3421 3429); do
  for spec in "ls100 80 100" "ns100 80 100" "fsc89 69 89"; do
    read -r dataset base total <<<"${spec}"
    feature_dir="artifacts/vbcgcd_features/${dataset}/seed3420"
    python third_party/OFCL/run_audio_features.py \
      --feature-dir "${feature_dir}" --output "artifacts/ofcl_transfer/${dataset}/seed${seed}" \
      --base "${base}" --increment 5 --sessions 5 --seed "${seed}"
    python scripts/run_opcr_features.py \
      --feature-dir "${feature_dir}" --output "artifacts/opcr/${dataset}/seed${seed}" \
      --base "${base}" --total "${total}" --increment 5 --sessions 5 --seed "${seed}"
    python scripts/run_yloc_features.py \
      --feature-dir "${feature_dir}" --output "artifacts/yloc/${dataset}/seed${seed}" \
      --base "${base}" --increment 5 --sessions 5 --seed "${seed}"
  done
done
