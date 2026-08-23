#!/usr/bin/env bash
set -euo pipefail

# GPU 1 is occupied by the formal NS reliability-gate run. Queue the joint
# ablation behind that exact process without reserving GPU memory while waiting.
while kill -0 2769809 2>/dev/null; do
  sleep 20
done

cd /data/lqq/baseline_dfsb
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=1 python train_unopenset.py \
  -config configs/exp_ls100_dfsb_eval.yml \
  -dataset librispeech --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True --checkpoint_name epoch_15.pth \
  --num_labeled_classes 80 --num_unlabeled_classes 5 \
  --opt_version proposed --run_tag joint_dfsb_umr_ls_screen \
  --save_result /data/lqq/baseline_dfsb/save_result/ \
  --mixed_openworld_stream True \
  --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
  --structure_discovery_weight 0.5 --cluster_all_candidates False \
  --compact_base_margin 0.2 --compact_novel_margin 0.0 \
  --stat_memory True --stat_base_var 0.000290403 --stat_cov_shrink 0.7 \
  --stat_replay_samples 16 --stat_replay_steps 30 --stat_replay_lr 0.03 \
  --stat_anchor_weight 2.0 --stat_update_strength 0.1 \
  --stat_old_update_strength 0.0 --stat_min_base_acc 0.7 \
  --reset_fc_each_round True --eval_repeats 1 \
  2>&1 | tee logs/proposed_joint_dfsb_umr_ls_screen.log
