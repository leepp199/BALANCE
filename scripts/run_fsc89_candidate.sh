#!/usr/bin/env bash
set -euo pipefail
bias="${1:--0.02}"
reflow="${2:-0.1}"
tag="${3:-candidate}"
repeats="${4:-1}"
osr_scale="${5:-1.0}"
bank_topk="${6:-3}"
joint_weight="${7:-0.5}"
joint_layer="${8:-layer4}"
tta_views="${9:-1}"
support_blend="${10:-1.0}"
hubness_weight="${11:-0.0}"
hubness_scope="${12:-all}"
novel_scale="${13:-1.0}"
stat_memory="${14:-False}"
stat_min_base_acc="${15:-0.7}"
stat_min_variance="${16:-1e-4}"
session_align="${17:-True}"
group_gate="${18:-False}"
bank_temp="${19:-0.0}"
bank_classifier="${20:-True}"
radius_power="${21:-0.0}"
base_scale="${22:-1.0}"
bank_blend="${23:-1.0}"
osr_group_gate="${24:-False}"
cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/data/lqq/baseline_dfsb/.offline_torch
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_unopenset.py \
  -config configs/exp_fsc89.yml -dataset FMC \
  --dataroot /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data \
  --checkpoint True --full_checkpoint_path artifacts/fsc_episodic_encoder_aug_l4.pth \
  --skip_meta_train True --skip_replace_base_fc False --eval_repeats "$repeats" --reset_fc_each_round True \
  --mixed_openworld_stream True --cluster_all_candidates False \
  --discovery_ranked_topk 25 --discovery_rank_score encode_joint_cosine \
  --base_geometry_path artifacts/fsc_episodic_encoder_aug_l4_geometry.pth \
  --discovery_encoder direct --encode_tta_views "$tta_views" --normalize_cluster_features True \
  --use_joint_cluster_assignments True --joint_cluster_layer "$joint_layer" --joint_margin_weight "$joint_weight" \
  --joint_kmeans_trials 1 --cluster_algorithm kmeans --prototype_trim_farthest 0 \
  --compact_steps 0 --compact_base_margin 0.2 --compact_novel_margin 0.0 \
  --novel_base_projection_strength 0.0 --balanced_kmeans True --incremental_metric cosine \
  --incremental_novel_logit_bias "$bias" --incremental_novel_logit_scale "$novel_scale" --incremental_base_logit_scale "$base_scale" --incremental_proto_hubness_weight "$hubness_weight" --incremental_proto_hubness_scope "$hubness_scope" --incremental_proto_hubness_k 8 \
  --incremental_group_margin_gate "$group_gate" --incremental_group_margin_bias 0.0 \
  --incremental_osr_group_gate "$osr_group_gate" --incremental_group_router_path '' \
  --incremental_group_router_offset 0.0 --incremental_group_router_soft_scale 0.0 \
  --incremental_tree_router_path '' --incremental_tree_router_soft_scale 0.0 \
  --incremental_radius_power "$radius_power" --oracle_eval_group_gate False \
  --incremental_quantile_group_gate False --incremental_quantile_support_topk 1 \
  --incremental_quantile_score support_margin --incremental_sinkhorn_balance False \
  --incremental_sinkhorn_temperature 0.05 --incremental_sinkhorn_iterations 100 \
  --incremental_sinkhorn_scope class --discovery_reflow_quantile "$reflow" --discovery_reflow_max 8 \
  --osr_thr_scale "$osr_scale" \
  --support_proto_blend "$support_blend" \
  --stat_memory "$stat_memory" \
  --stat_min_base_acc "$stat_min_base_acc" --stat_min_variance "$stat_min_variance" \
  --novel_bank_classifier "$bank_classifier" --session_restricted_alignment "$session_align" --novel_bank_topk "$bank_topk" --novel_bank_temperature "$bank_temp" --novel_bank_blend "$bank_blend" \
  --oracle_cluster False --run_tag "fsc89_candidate_${tag}" 2>&1 | tee "logs/fsc89_candidate_${tag}.log"
