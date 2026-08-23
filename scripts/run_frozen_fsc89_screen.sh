#!/usr/bin/env bash
set -euo pipefail
novel_bias="${1:-0.0}"
compact_steps="${2:-0}"
base_margin="${3:-0.2}"
projection_strength="${4:-0.0}"
checkpoint_path="${5:-/data/lqq/baseline/save/exp_fsc89_69supcon/base_train_for_meta_FMC.pth}"
skip_replace="${6:-True}"
geometry_path="${7:-/data/lqq/baseline/save/exp_fsc89_69supcon/base_val_geometry_features.pth}"
oracle_cluster="${8:-False}"
hubness_weight="${9:-0.0}"
hubness_k="${10:-8}"
incremental_metric="${11:-cosine}"
linear_adapter_path="${12:-}"
linear_adapter_strength="${13:-1.0}"
group_margin_gate="${14:-False}"
group_margin_bias="${15:-0.0}"
novel_bank_classifier="${16:-False}"
novel_bank_topk="${17:-1}"
cluster_all_candidates="${18:-True}"
osr_group_gate="${19:-False}"
group_router_path="${20:-}"
group_router_offset="${21:-0.0}"
group_router_soft_scale="${22:-0.0}"
tree_router_path="${23:-}"
tree_router_soft_scale="${24:-0.0}"
radius_power="${25:-0.0}"
oracle_eval_group_gate="${26:-False}"
quantile_group_gate="${27:-False}"
quantile_support_topk="${28:-1}"
quantile_score="${29:-support_margin}"
sinkhorn_balance="${30:-False}"
sinkhorn_temperature="${31:-0.05}"
sinkhorn_iterations="${32:-100}"
sinkhorn_scope="${33:-class}"
reflow_quantile="${34:-0.0}"
reflow_max="${35:-10}"
eval_repeats="${36:-1}"
run_suffix="${37:-}"
cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/lqq/baseline_dfsb/.offline_torch
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_unopenset.py \
  -config configs/exp_fsc89.yml -dataset FMC \
  --dataroot /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data \
  --checkpoint True \
  --full_checkpoint_path "$checkpoint_path" \
  --skip_meta_train True --skip_replace_base_fc "$skip_replace" \
  --eval_repeats "$eval_repeats" --reset_fc_each_round True \
  --mixed_openworld_stream True --cluster_all_candidates "$cluster_all_candidates" \
  --discovery_ranked_topk 25 --discovery_rank_score encode_joint_cosine \
  --base_geometry_path "$geometry_path" \
  --discovery_encoder direct --encode_tta_views 1 \
  --normalize_cluster_features True --use_joint_cluster_assignments True \
  --joint_cluster_layer layer4 --joint_margin_weight 0.5 --joint_kmeans_trials 1 \
  --cluster_algorithm kmeans --prototype_trim_farthest 0 --compact_steps "$compact_steps" \
  --prototype_linear_adapter_path "$linear_adapter_path" \
  --prototype_linear_adapter_strength "$linear_adapter_strength" \
  --compact_base_margin "$base_margin" --compact_novel_margin 0.0 \
  --novel_base_projection_strength "$projection_strength" \
  --balanced_kmeans True --incremental_metric "$incremental_metric" \
  --incremental_novel_logit_bias "$novel_bias" \
  --incremental_proto_hubness_weight "$hubness_weight" \
  --incremental_proto_hubness_k "$hubness_k" \
  --incremental_group_margin_gate "$group_margin_gate" \
  --incremental_group_margin_bias "$group_margin_bias" \
  --incremental_osr_group_gate "$osr_group_gate" \
  --incremental_group_router_path "$group_router_path" \
  --incremental_group_router_offset "$group_router_offset" \
  --incremental_group_router_soft_scale "$group_router_soft_scale" \
  --incremental_tree_router_path "$tree_router_path" \
  --incremental_tree_router_soft_scale "$tree_router_soft_scale" \
  --incremental_radius_power "$radius_power" \
  --oracle_eval_group_gate "$oracle_eval_group_gate" \
  --incremental_quantile_group_gate "$quantile_group_gate" \
  --incremental_quantile_support_topk "$quantile_support_topk" \
  --incremental_quantile_score "$quantile_score" \
  --incremental_sinkhorn_balance "$sinkhorn_balance" \
  --incremental_sinkhorn_temperature "$sinkhorn_temperature" \
  --incremental_sinkhorn_iterations "$sinkhorn_iterations" \
  --incremental_sinkhorn_scope "$sinkhorn_scope" \
  --discovery_reflow_quantile "$reflow_quantile" \
  --discovery_reflow_max "$reflow_max" \
  --novel_bank_classifier "$novel_bank_classifier" --session_restricted_alignment True \
  --novel_bank_topk "$novel_bank_topk" \
  --oracle_cluster "$oracle_cluster" \
  --run_tag "fsc89_${incremental_metric}_bias_${novel_bias//./p}_bank${novel_bank_classifier}_top${novel_bank_topk}_clusterall${cluster_all_candidates}_ogate${oracle_eval_group_gate}_qgate${quantile_group_gate}_qtop${quantile_support_topk}_${quantile_score}_sink${sinkhorn_balance}_${sinkhorn_scope}_t${sinkhorn_temperature}_reflow${reflow_quantile}${run_suffix}" \
  2>&1 | tee "logs/fsc89_${incremental_metric}_bias_${novel_bias//./p}_bank${novel_bank_classifier}_top${novel_bank_topk}_clusterall${cluster_all_candidates}_ogate${oracle_eval_group_gate}_qgate${quantile_group_gate}_qtop${quantile_support_topk}_${quantile_score}_sink${sinkhorn_balance}_${sinkhorn_scope}_t${sinkhorn_temperature}_reflow${reflow_quantile}${run_suffix}.log"
