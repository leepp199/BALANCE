#!/usr/bin/env bash
set -euo pipefail

cd /data/lqq/baseline_dfsb

wait_for_pid() {
  local target_pid="$1"
  while kill -0 "$target_pid" 2>/dev/null; do
    sleep 20
  done
}

case "${1:?usage: $0 ls-ds|ls-umr|fsc-bcd}" in
  ls-ds)
    wait_for_pid 2842239
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python train_unopenset.py \
      -config configs/exp_ls100_dfsb_eval.yml -dataset librispeech \
      --dataroot /data/datasets/librispeech_fscil/ \
      --checkpoint True --checkpoint_name epoch_15.pth \
      --num_labeled_classes 80 --num_unlabeled_classes 5 \
      --opt_version raw_v2 --run_tag dfsb_ds_ls100_10runs \
      --save_result /data/lqq/baseline_dfsb/save_result/ \
      --mixed_openworld_stream True \
      --structure_discovery_checkpoint artifacts/checkpoints/dfsb_ls100_base.pth \
      --structure_discovery_weight 0.5 --cluster_all_candidates False \
      --teen_calibration False --stat_memory False \
      --reset_fc_each_round True --eval_repeats 10 \
      2>&1 | tee logs/raw_v2_dfsb_ds_ls100_10runs.log
    ;;
  ls-umr)
    wait_for_pid 2842875
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python train_unopenset.py \
      -config configs/exp_ls100.yml -dataset librispeech \
      --dataroot /data/datasets/librispeech_fscil/ \
      --checkpoint True --checkpoint_name ../backup_epoch_1777217149/epoch_15.pth \
      --num_labeled_classes 80 --num_unlabeled_classes 5 \
      --opt_version raw_v2 --run_tag umr_ls100_10runs \
      --save_result /data/lqq/baseline_dfsb/save_result/ \
      --mixed_openworld_stream True --structure_discovery_weight 0.0 \
      --cluster_all_candidates False \
      --teen_calibration True --teen_alpha 0.9 --teen_topk 0 \
      --teen_temperature 0.0625 --teen_preserve_norm False \
      --stat_memory True --stat_base_var 0.000290403 --stat_cov_shrink 0.7 \
      --stat_replay_samples 16 --stat_replay_steps 30 --stat_replay_lr 0.03 \
      --stat_anchor_weight 2.0 --stat_update_strength 0.1 \
      --stat_old_update_strength 0.0 --stat_min_base_acc 0.7 \
      --reset_fc_each_round True --eval_repeats 10 \
      2>&1 | tee logs/raw_v2_umr_ls100_10runs.log
    ;;
  fsc-bcd)
    wait_for_pid 2843611
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python train_unopenset.py \
      -config configs/exp_fsc89.yml -dataset FMC \
      --dataroot /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data \
      --checkpoint True --checkpoint_name epoch_15.pth \
      --num_labeled_classes 69 --num_unlabeled_classes 5 \
      --opt_version raw_v2 --run_tag bcd_fsc89_10runs \
      --save_result /data/lqq/baseline_dfsb/save_result/ \
      --mixed_openworld_stream True --structure_discovery_weight 0.0 \
      --cluster_all_candidates False \
      --teen_calibration True --teen_alpha 0.9 --teen_topk 0 \
      --teen_temperature 0.0625 --teen_preserve_norm False \
      --stat_memory False --balanced_kmeans True --balanced_kmeans_iters 5 \
      --reset_fc_each_round True --eval_repeats 10 \
      2>&1 | tee logs/raw_v2_bcd_fsc89_10runs.log
    ;;
esac
