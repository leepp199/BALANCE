#!/bin/bash
# Run NS-100 Full Pipeline Experiment
set -e
BASE_DIR="/data/lqq/baseline"
PYTHON="/data/lqq/miniconda3/bin/python"
export CUDA_VISIBLE_DEVICES=0

echo "=== NS-100 Full Pipeline @ $(date) ==="

cd ${BASE_DIR}
${PYTHON} train_unopenset.py \
  -config configs/exp_ns100.yml \
  --dataset nsynth-100 \
  --dataroot /data/datasets/The_NSynth_Dataset \
  --pretrained_model_path ${BASE_DIR}/save/base_train_for_meta_nsynth-100.pth \
  --load_base True \
  --checkpoint False \
  --num_labeled_classes 80 \
  --num_unlabeled_classes 5 \
  --opt_version final_exp \
  --run_tag ns100_full \
  --save_result ${BASE_DIR}/save_result/ \
  --train_weight_base 1 \
  --base_seman_calib 1 \
  --hinge_margin 2.0 \
  --osr_noise_std 0.1 \
  --train_noise_std 0.1 \
  --osr_thr_scale 1.0 \
  2>&1 | tee ${BASE_DIR}/save_result/final_exp/ns100_full.log

echo "=== NS-100 Done @ $(date) ==="
