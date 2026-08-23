#!/usr/bin/env bash
set -euo pipefail
cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/lqq/baseline_dfsb/.offline_torch
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_unopenset.py \
  -config configs/exp_ns100.yml -dataset nsynth-100 \
  --dataroot /data/datasets/The_NSynth_Dataset \
  --checkpoint True --checkpoint_name epoch_15.pth --skip_meta_train True \
  --eval_repeats 50 --reset_fc_each_round True \
  --mixed_openworld_stream True --cluster_all_candidates True \
  --kmeans_filter_quantile 0.5 --balanced_kmeans True \
  --balanced_kmeans_iters 5 --run_tag ns_all_q50_cana_50final_repro \
  2>&1 | tee logs/ns_all_q50_cana_50final_repro.log

