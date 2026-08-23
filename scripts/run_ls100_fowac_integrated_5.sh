#!/usr/bin/env bash
set -euo pipefail

cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Unified FOWAC evaluation: LSRB + CANA + model.encode prototypes + UMR.
python train_unopenset.py \
  -config configs/exp_ls100_dfsb_eval.yml \
  -dataset librispeech --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True --checkpoint_name epoch_15.pth \
  --num_labeled_classes 80 --num_unlabeled_classes 5 \
  --opt_version raw_v2 --run_tag fowac_integrated_ls100_corrected_5 \
  --save_result /data/lqq/baseline_dfsb/save_result/ \
  --mixed_openworld_stream True \
  --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
  --structure_discovery_weight 0.5 --cluster_all_candidates False \
  --balanced_kmeans True --balanced_kmeans_iters 5 \
  --compact_base_margin 0.2 --compact_novel_margin 0.0 \
  --teen_calibration False \
  --stat_memory True --stat_base_var 0.000290403 --stat_cov_shrink 0.7 \
  --stat_replay_samples 16 --stat_replay_steps 30 --stat_replay_lr 0.03 \
  --stat_anchor_weight 2.0 --stat_update_strength 0.1 \
  --stat_old_update_strength 0.0 --stat_min_base_acc 0.7 \
  --session_restricted_alignment True \
  --reset_fc_each_round True --eval_repeats 5 \
  2>&1 | tee logs/ls100_fowac_integrated_corrected_5.log
