#!/bin/bash
# Final Experiment Runner - NS-100 & FSC-89 & LS-100
set -e

BASE_DIR="/data/lqq/baseline"
PYTHON="/data/lqq/miniconda3/bin/python"
SAVE_RESULT="${BASE_DIR}/save_result/final_exp"
mkdir -p ${SAVE_RESULT}

# ============================================================
# Experiment 1: NS-100 Full Pipeline  
# ============================================================
echo "============================================================"
echo "[1/3] NS-100 Full Pipeline @ $(date)"
echo "============================================================"

cd ${BASE_DIR}
CUDA_VISIBLE_DEVICES=0 ${PYTHON} train_unopenset.py \
  -config configs/exp_ns100.yml \
  --dataset nsynth-100 \
  --dataroot /data/datasets/The_NSynth_Dataset \
  --pretrained_model_path ${BASE_DIR}/save/base_train_for_meta_nsynth-100.pth \
  --load_base True \
  --checkpoint False \
  --num_labeled_classes 80 \
  --num_unlabeled_classes 5 \
  --opt_version final_exp \
  --run_tag ns100 \
  --save_result ${BASE_DIR}/save_result/ \
  --train_weight_base 1 \
  --hinge_margin 2.0 \
  --osr_noise_std 0.1 \
  --train_noise_std 0.1 \
  2>&1 | tee ${SAVE_RESULT}/ns100_full.log

echo "NS-100 done @ $(date)"

# ============================================================
# Experiment 2: FSC-89 Full Pipeline
# ============================================================
echo "============================================================"
echo "[2/3] FSC-89 Full Pipeline @ $(date)"
echo "============================================================"

cd ${BASE_DIR}
CUDA_VISIBLE_DEVICES=1 ${PYTHON} train_unopenset.py \
  -config configs/exp_fsc89.yml \
  --dataset FMC \
  --dataroot /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data \
  --pretrained_model_path ${BASE_DIR}/save/base_train_for_meta_FMC.pth \
  --load_base True \
  --checkpoint False \
  --num_labeled_classes 69 \
  --num_unlabeled_classes 5 \
  --opt_version final_exp \
  --run_tag fsc89 \
  --save_result ${BASE_DIR}/save_result/ \
  --train_weight_base 1 \
  --hinge_margin 2.0 \
  --osr_noise_std 0.1 \
  --train_noise_std 0.1 \
  2>&1 | tee ${SAVE_RESULT}/fsc89_full.log

echo "FSC-89 done @ $(date)"

# ============================================================
# Experiment 3: LS-100 Full Pipeline  
# ============================================================
echo "============================================================"
echo "[3/3] LS-100 Full Pipeline @ $(date)"
echo "============================================================"

cd ${BASE_DIR}
CUDA_VISIBLE_DEVICES=2 ${PYTHON} train_unopenset.py \
  -config configs/exp_ls100.yml \
  -dataset librispeech \
  --dataroot /data/datasets/librispeech_fscil/ \
  --checkpoint True \
  --checkpoint_name ../backup_epoch_1777217149/epoch_15.pth \
  --num_labeled_classes 80 \
  --num_unlabeled_classes 5 \
  --opt_version final_exp \
  --run_tag ls100 \
  --save_result ${BASE_DIR}/save_result/ \
  --train_weight_base 1 \
  --hinge_margin 2.0 \
  --osr_noise_std 0.1 \
  --train_noise_std 0.1 \
  --old_proto_adapt False \
  2>&1 | tee ${SAVE_RESULT}/ls100_full.log

echo "LS-100 done @ $(date)"
echo "============================================================"
echo "All experiments completed @ $(date)"
echo "============================================================"
