#!/usr/bin/env bash
set -euo pipefail
cd /data/lqq/baseline_dfsb
for seed in $(seq 3421 3429); do
  for spec in "ls100 80" "ns100 80" "fsc89 69"; do
    read -r dataset base <<<"${spec}"
    CUDA_VISIBLE_DEVICES=2 python scripts/run_happy_features.py \
      --feature-dir "artifacts/vbcgcd_features/${dataset}/seed3420" \
      --output "artifacts/happy/${dataset}/seed${seed}" --base "${base}" \
      --seed "${seed}" --steps 100 --batch-size 256 --device cuda
  done
done
