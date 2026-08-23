#!/usr/bin/env bash
set -euo pipefail

while kill -0 2831185 2>/dev/null; do
  sleep 20
done

cd /data/lqq/baseline_dfsb
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=1 python train_unopenset.py \
  -config configs/exp_ns100.yml -dataset nsynth-100 \
  --dataroot /data/datasets/The_NSynth_Dataset/ \
  --checkpoint True --checkpoint_name epoch_15.pth \
  --num_labeled_classes 80 --num_unlabeled_classes 5 \
  --opt_version proposed --run_tag discovery_reflow_q20_ns_screen \
  --save_result /data/lqq/baseline_dfsb/save_result/ \
  --mixed_openworld_stream True --structure_discovery_weight 0.0 \
  --cluster_all_candidates False --discovery_reflow_quantile 0.20 \
  --discovery_reflow_max 8 \
  --teen_calibration True --teen_alpha 0.9 --teen_topk 0 \
  --teen_temperature 0.0625 --teen_preserve_norm False \
  --stat_memory False --reset_fc_each_round True --eval_repeats 1 \
  2>&1 | tee logs/proposed_discovery_reflow_q20_ns_screen.log
