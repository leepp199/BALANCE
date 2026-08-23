#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/lqq/baseline"
PYTHON="/data/lqq/miniconda3/bin/python"
GPU_ID="${GPU_ID:-2}"

cd "${BASE_DIR}"
mkdir -p "${BASE_DIR}/save_result/final_exp_repro_ckpt"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" train_unopenset.py \
  -config configs/exp_ls100.yml \
  -dataset librispeech \
  --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True \
  --checkpoint_name ../backup_epoch_1777217149/epoch_15.pth \
  --num_labeled_classes 80 \
  --num_unlabeled_classes 5 \
  --opt_version final_exp_repro_ckpt \
  --run_tag ls100 \
  --save_result "${BASE_DIR}/save_result/" \
  --train_weight_base 1 \
  --hinge_margin 2.0 \
  --osr_noise_std 0.1 \
  --train_noise_std 0.1 \
  --old_proto_adapt False \
  --joint_proto_refine False \
  --proto_ema_alpha 0.0 \
  --kmeans_filter_thr 0.0 \
  --reset_fc_each_round False \
  2>&1 | tee "${BASE_DIR}/save_result/final_exp_repro_ckpt/ls100_repro_ckpt.log"
