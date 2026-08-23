#!/usr/bin/env bash
set -euo pipefail

cd /data/lqq/baseline_dfsb
screen_pid="${1:-2370503}"
while kill -0 "${screen_pid}" 2>/dev/null; do
  sleep 10
done

CUDA_VISIBLE_DEVICES=0 python train_unopenset.py \
  -config configs/exp_ls100_dfsb_eval.yml \
  -dataset librispeech \
  --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True \
  --checkpoint_name epoch_15.pth \
  --num_labeled_classes 80 \
  --num_unlabeled_classes 5 \
  --opt_version dual_structure \
  --run_tag weight15_buffer_50runs \
  --save_result /data/lqq/baseline_dfsb/save_result/ \
  --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
  --structure_discovery_weight 1.5 \
  --cluster_all_candidates True \
  --compact_base_margin 0.2 \
  --compact_novel_margin 0.0 \
  --old_proto_adapt False \
  --joint_proto_refine False \
  --proto_ema_alpha 0.0 \
  --kmeans_filter_thr 0.0 \
  --reset_fc_each_round True \
  --eval_repeats 50 \
  > logs/dual_structure_w15_buffer_50runs.log 2>&1

touch artifacts/BACKGROUND_EXPERIMENT_COMPLETE
