#!/usr/bin/env bash
set -euo pipefail

cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/lqq/baseline_dfsb/.offline_torch
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Frozen after a five-stream screening mean of 72.40% AA-inc.
# LS reliability routing keeps LSRB active and disables CANA/UMR, which were
# negative on the screening split. Discovery and prototypes use model.encode.
python train_unopenset.py \
  -config configs/exp_ls100_dfsb_eval.yml \
  -dataset librispeech --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True --checkpoint_name epoch_15.pth \
  --num_labeled_classes 80 --num_unlabeled_classes 5 \
  --opt_version raw_v2 --run_tag fowac_ls100_direct_frozen_50 \
  --save_result /data/lqq/baseline_dfsb/save_result/ \
  --mixed_openworld_stream True --discovery_encoder direct \
  --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
  --structure_discovery_weight 0.5 --cluster_all_candidates False \
  --balanced_kmeans False --normalize_cluster_features True \
  --compact_steps 30 --compact_base_margin 0.2 --compact_novel_margin 0.0 \
  --teen_calibration False --stat_memory False \
  --novel_bank_classifier False --use_pan_incremental False \
  --session_restricted_alignment True --oracle_cluster False \
  --reset_fc_each_round True --eval_repeats 50 \
  2>&1 | tee logs/ls100_fowac_direct_frozen_50.log
