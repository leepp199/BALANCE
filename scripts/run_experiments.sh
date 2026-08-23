#!/bin/bash
# Experiment Runner Script for Baseline Diagnostics & Fixes
# Creates copies for experiments, leaves original source untouched

set -e

BASE_DIR="/data/lqq/baseline"
PYTHON="/data/lqq/miniconda3/bin/python"

echo "========================================="
echo "Starting Baseline Experiments"
echo "Date: $(date)"
echo "========================================="

# ==============================================
# Experiment 1: NS-100 Full Pipeline
# Uses nsynth-100 base model, meta-train, eval
# ==============================================
echo ""
echo "========== Experiment 1: NS-100 =========="
echo "Log: ${BASE_DIR}/save_result/final_exp/ns100_full.log"

cd ${BASE_DIR}
${PYTHON} train_unopenset.py \
  -config configs/exp_ns100.yml \
  --dataset nsynth-100 \
  --dataroot /data/datasets/The_NSynth_Dataset \
  --pretrained_model_path ${BASE_DIR}/save/base_train_for_meta_nsynth-100.pth \
  --load_base True \
  --checkpoint False \
  --opt_version final_exp \
  --run_tag ns100_full \
  --save_result ${BASE_DIR}/save_result/ \
  --num_labeled_classes 80 \
  --num_unlabeled_classes 5 \
  2>&1 | tee ${BASE_DIR}/save_result/final_exp/ns100_full.log

echo "NS-100 done at $(date)"

EOF
