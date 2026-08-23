#!/usr/bin/env bash
set -euo pipefail
cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/lqq/baseline_dfsb/.offline_torch
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_unopenset.py \
  -config configs/exp_fsc89.yml -dataset FMC \
  --dataroot /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data \
  --checkpoint True \
  --full_checkpoint_path /data/lqq/baseline/save/exp_fsc89_69supcon/base_train_for_meta_FMC.pth \
  --skip_meta_train True --skip_replace_base_fc True \
  --eval_repeats 1 --reset_fc_each_round True \
  --mixed_openworld_stream True --cluster_all_candidates True \
  --discovery_ranked_topk 25 --discovery_rank_score encode_joint_cosine \
  --base_geometry_path /data/lqq/baseline/save/exp_fsc89_69supcon/base_val_geometry.pth \
  --discovery_encoder direct --encode_tta_views 1 \
  --normalize_cluster_features True --use_joint_cluster_assignments True \
  --joint_cluster_layer layer4 --joint_margin_weight 0.5 --joint_kmeans_trials 1 \
  --joint_kmeans_random_state legacy_none --cluster_algorithm kmeans \
  --compact_steps 0 --balanced_kmeans True --incremental_metric euclidean \
  --novel_bank_classifier True --session_restricted_alignment False \
  --run_tag fsc89_5765_historical_audit \
  2>&1 | tee logs/fsc89_5765_historical_audit.log
