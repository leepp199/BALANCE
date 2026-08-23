#!/usr/bin/env bash
set -euo pipefail

cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python train_unopenset.py \
  -config configs/exp_ls100.yml -dataset librispeech \
  --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True --checkpoint_name ../backup_epoch_1777217149/epoch_15.pth \
  --num_labeled_classes 80 --num_unlabeled_classes 5 \
  --opt_version raw_v2 --run_tag umr_ls100_corrected_5 \
  --save_result /data/lqq/baseline_dfsb/save_result/ \
  --mixed_openworld_stream True --structure_discovery_weight 0.0 \
  --cluster_all_candidates False \
  --teen_calibration True --teen_alpha 0.9 --teen_topk 0 \
  --teen_temperature 0.0625 --teen_preserve_norm False \
  --stat_memory True --stat_base_var 0.000290403 --stat_cov_shrink 0.7 \
  --stat_replay_samples 16 --stat_replay_steps 30 --stat_replay_lr 0.03 \
  --stat_anchor_weight 2.0 --stat_update_strength 0.1 \
  --stat_old_update_strength 0.0 --stat_min_base_acc 0.7 \
  --session_restricted_alignment True \
  --reset_fc_each_round True --eval_repeats 5 \
  2>&1 | tee logs/ls100_umr_corrected_5.log
