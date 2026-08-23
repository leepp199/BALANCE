#!/usr/bin/env bash
set -euo pipefail

while pgrep -f 'run_tag teen_a09_dual05_10runs' >/dev/null; do
  sleep 20
done

cd /data/lqq/baseline_dfsb
CUDA_VISIBLE_DEVICES=0 python train_unopenset.py \
  -config configs/exp_ls100_dfsb_eval.yml -dataset librispeech \
  --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True --checkpoint_name epoch_15.pth \
  --num_labeled_classes 80 --num_unlabeled_classes 5 \
  --opt_version mixed_openworld --run_tag teen_official_transfer_10runs \
  --save_result /data/lqq/baseline_dfsb/save_result/ \
  --mixed_openworld_stream True \
  --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
  --structure_discovery_weight 0.5 --cluster_all_candidates False \
  --compact_base_margin 0.2 --compact_novel_margin 0.0 \
  --teen_calibration True --teen_alpha 0.9 --teen_topk 0 \
  --teen_temperature 0.0625 --teen_preserve_norm False \
  --old_proto_adapt False --joint_proto_refine False --proto_ema_alpha 0.0 \
  --kmeans_filter_thr 0.0 --reset_fc_each_round True --eval_repeats 10 \
  2>&1 | tee logs/mixed_openworld/teen_official_transfer_10runs.log
