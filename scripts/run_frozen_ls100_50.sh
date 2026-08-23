#!/usr/bin/env bash
set -euo pipefail
cd /data/lqq/baseline_dfsb
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/data/lqq/baseline_dfsb/.offline_torch
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python train_unopenset.py \
  -config configs/exp_ls100_dfsb_eval.yml -dataset librispeech \
  --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True --checkpoint_name epoch_15.pth --skip_meta_train True \
  --num_labeled_classes 80 --num_unlabeled_classes 5 \
  --mixed_openworld_stream True \
  --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
  --structure_discovery_weight 0.5 --cluster_all_candidates False \
  --teen_calibration False --stat_memory False \
  --reset_fc_each_round True --eval_repeats 50 \
  --run_tag ls100_lsrb_50final \
  2>&1 | tee logs/ls100_lsrb_50final.log

