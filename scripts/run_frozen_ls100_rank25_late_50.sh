#!/usr/bin/env bash
set -euo pipefail
cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/lqq/baseline_dfsb/.offline_torch
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python train_unopenset.py \
  -config configs/exp_ls100_dfsb_eval.yml -dataset librispeech \
  --dataroot /data/datasets/librispeech_fscil/ --checkpoint True --checkpoint_name epoch_15.pth \
  --num_labeled_classes 80 --num_unlabeled_classes 5 --opt_version raw_v2 \
  --mixed_openworld_stream True --discovery_encoder direct \
  --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
  --structure_discovery_weight 0.5 --cluster_all_candidates True \
  --discovery_ranked_topk 25 --discovery_rank_score encode_maxlogit \
  --discovery_rank_start_session 3 --balanced_kmeans True --balanced_kmeans_iters 5 \
  --normalize_cluster_features True --compact_steps 30 --compact_base_margin 0.2 \
  --compact_novel_margin 0.0 --stat_memory False --teen_calibration False \
  --novel_bank_classifier False --session_restricted_alignment True \
  --reset_fc_each_round True --eval_repeats 50 --run_tag ls100_rank25_late_frozen_50 \
  2>&1 | tee logs/ls100_rank25_late_frozen_50.log
