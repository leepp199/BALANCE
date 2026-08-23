#!/usr/bin/env bash
set -euo pipefail

scale="${1:?usage: $0 OSR_THRESHOLD_SCALE [REPEATS]}"
repeats="${2:-3}"
filter_quantile="${3:-0.0}"
base_var="${4:-0.000290403}"
compact_steps="${5:-30}"
adapter_path="${6:-}"
semantic_calibration="${7:-False}"
normalize_cluster="${8:-False}"
discovery_encoder="${9:-direct}"
novel_bank_classifier="${10:-False}"
use_pan="${11:-False}"
balanced="${12:-True}"
stat_memory="${13:-True}"
oracle="${14:-False}"
tag="${scale//./p}_q${filter_quantile//./p}_v${base_var//./p}_c${compact_steps}"\
"_enc${discovery_encoder}_bank${novel_bank_classifier}_pan${use_pan}"\
"_bal${balanced}_stat${stat_memory}_oracle${oracle}"
cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python train_unopenset.py \
  -config configs/exp_ls100_dfsb_eval.yml \
  -dataset librispeech --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True --checkpoint_name epoch_15.pth \
  --num_labeled_classes 80 --num_unlabeled_classes 5 \
  --opt_version raw_v2 --run_tag "fowac_ls100_gate_${tag}_${repeats}" \
  --save_result /data/lqq/baseline_dfsb/save_result/ \
  --mixed_openworld_stream True \
  --discovery_encoder "$discovery_encoder" \
  --novel_bank_classifier "$novel_bank_classifier" \
  --use_pan_incremental "$use_pan" \
  --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
  --structure_discovery_weight 0.5 --cluster_all_candidates False \
  --balanced_kmeans "$balanced" --balanced_kmeans_iters 5 \
  --normalize_cluster_features "$normalize_cluster" \
  --kmeans_filter_quantile "$filter_quantile" \
  --compact_steps "$compact_steps" --compact_base_margin 0.2 --compact_novel_margin 0.0 \
  --robust_proto_adapter_path "$adapter_path" \
  --teen_calibration "$semantic_calibration" --teen_alpha 0.9 --teen_topk 0 \
  --teen_temperature 0.0625 --teen_preserve_norm False \
  --osr_thr_scale "$scale" \
  --stat_memory "$stat_memory" --stat_base_var "$base_var" --stat_cov_shrink 0.7 \
  --stat_replay_samples 16 --stat_replay_steps 30 --stat_replay_lr 0.03 \
  --stat_anchor_weight 2.0 --stat_update_strength 0.1 \
  --stat_old_update_strength 0.0 --stat_min_base_acc 0.7 \
  --session_restricted_alignment True \
  --oracle_cluster "$oracle" \
  --reset_fc_each_round True --eval_repeats "$repeats" \
  2>&1 | tee "logs/ls100_fowac_gate_${tag}_${repeats}.log"
