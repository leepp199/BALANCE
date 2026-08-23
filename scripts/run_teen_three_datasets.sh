#!/usr/bin/env bash
set -euo pipefail
cd /data/lqq/baseline_dfsb
mkdir -p logs/sota_three_datasets save_result/sota_three_datasets

run_one() {
  local dataset="$1" config="$2" root="$3" base="$4" tag="$5"
  CUDA_VISIBLE_DEVICES=0 python train_unopenset.py \
    -config "$config" -dataset "$dataset" --dataroot "$root" \
    --checkpoint True --checkpoint_name epoch_15.pth \
    --num_labeled_classes "$base" --num_unlabeled_classes 5 \
    --opt_version sota_three_datasets --run_tag "$tag" \
    --save_result /data/lqq/baseline_dfsb/save_result/ \
    --mixed_openworld_stream True --structure_discovery_weight 0.0 \
    --cluster_all_candidates False --compact_base_margin 0.2 \
    --compact_novel_margin 0.0 --teen_calibration True \
    --teen_alpha 0.9 --teen_topk 0 --teen_temperature 0.0625 \
    --teen_preserve_norm False --old_proto_adapt False \
    --joint_proto_refine False --proto_ema_alpha 0.0 \
    --kmeans_filter_thr 0.0 --reset_fc_each_round True --eval_repeats 10 \
    2>&1 | tee "logs/sota_three_datasets/${tag}.log"
}

run_one librispeech configs/exp_ls100.yml \
  /data/datasets/librispeech_fscil/ 80 teen_pure_ls100_10runs
run_one nsynth-100 configs/exp_ns100.yml \
  /data/datasets/The_NSynth_Dataset/ 80 teen_pure_ns100_10runs
run_one FMC configs/exp_fsc89.yml \
  /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data 69 teen_pure_fsc89_10runs
