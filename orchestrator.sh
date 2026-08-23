#!/bin/bash
# Orchestrator: monitors current experiments, auto-launches remaining tasks
# Remaining tasks after current batch:
#   1. Ours NS-100 50x
#   2. Ours FSC-89 50x
#   3. FEC-OSL NS-100
#   4. FEC-OSL FSC-89
#   5. Generate comparison tables

PYTHON=/data/lqq/miniconda3/bin/python
BASE=/data/lqq/baseline

# Markers for task completion
LS_EVAL_MARKER=$BASE/save_result/baselines_ls_v2/comparison_table.csv
NS_EVAL_MARKER=$BASE/save_result/baselines_ns_v2/comparison_table.csv
FSC_EVAL_MARKER=$BASE/save_result/baselines_fsc89_v2/comparison_table.csv
OURS_LS_MARKER=$BASE/save_result/opt_v4/test_result.txt
FECOSL_LS_MARKER=$BASE/save_result/end_to_end/fec_osl_librispeech.txt

# Check if a CSV has meaningful content (>10 bytes, non-empty)
has_content() {
    [ -f "$1" ] && [ $(wc -c < "$1") -gt 10 ]
}

while true; do
    sleep 600  # Check every 10 min
    NOW=$(date "+%Y-%m-%d %H:%M:%S")
    echo "=== [$NOW] Orchestrator ==="

    # Check current running processes
    OURS_RUNNING=$(pgrep -f "train_unopenset.*ls100\|train_unopenset.*nsynth-100\|train_unopenset.*FMC" | wc -l)
    EVAL_RUNNING=$(pgrep -f "run_all_baselines" | wc -l)
    FECOSL_RUNNING=$(pgrep -f "run_fec_osl" | wc -l)

    echo "  Ours running: $OURS_RUNNING, Eval running: $EVAL_RUNNING, FEC-OSL running: $FECOSL_RUNNING"

    # --- Step 1: Launch Ours NS-100 50x after Ours LS completes AND baseline NS has some results ---
    if [ "$OURS_RUNNING" -eq 0 ] && [ ! -f "$BASE/save/ours_ns_running.flag" ]; then
        # Check if Ours LS results exist
        if [ -f "$OURS_LS_MARKER" ] && [ $(wc -c < "$OURS_LS_MARKER") -gt 10 ]; then
            echo "  >>> Launching Ours NS-100 50x..."
            cd $BASE
            touch $BASE/save/ours_ns_running.flag
            nohup $PYTHON train_unopenset.py \
                -config configs/exp_ns100.yml \
                -dataset nsynth-100 \
                --dataroot /data/datasets/nsynth-100/ \
                --pretrained_model_path $BASE/save/base_train_for_meta_nsynth-100.pth \
                --load_base True --checkpoint False \
                --num_labeled_classes 80 --num_unlabeled_classes 5 \
                --opt_version final_exp --run_tag ns100_50x \
                --save_result $BASE/save_result/ \
                --train_weight_base 1 --hinge_margin 2.0 \
                --osr_noise_std 0.1 --train_noise_std 0.1 \
                > $BASE/save/run_ours_ns50x.log 2>&1 &
            echo "  Ours NS PID: $!"
        fi
    fi

    # --- Step 2: Launch Ours FSC-89 50x after Ours LS completes ---
    if [ "$OURS_RUNNING" -eq 0 ] && [ ! -f "$BASE/save/ours_fsc_running.flag" ] && [ ! -f "$BASE/save/ours_ns_running.flag" ]; then
        if [ -f "$OURS_LS_MARKER" ] && [ $(wc -c < "$OURS_LS_MARKER") -gt 10 ]; then
            echo "  >>> Launching Ours FSC-89 50x..."
            cd $BASE
            touch $BASE/save/ours_fsc_running.flag
            nohup $PYTHON train_unopenset.py \
                -config configs/exp_fsc89.yml \
                -dataset FMC \
                --dataroot /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data \
                --pretrained_model_path $BASE/save/base_train_for_meta_FMC.pth \
                --load_base True --checkpoint False \
                --num_labeled_classes 69 --num_unlabeled_classes 5 \
                --opt_version final_exp --run_tag fsc89_50x \
                --save_result $BASE/save_result/ \
                --train_weight_base 1 --hinge_margin 2.0 \
                --osr_noise_std 0.1 --train_noise_std 0.1 \
                > $BASE/save/run_ours_fsc50x.log 2>&1 &
            echo "  Ours FSC89 PID: $!"
        fi
    fi

    # --- Step 3: FEC-OSL NS-100 after FEC-OSL LS completes ---
    if [ "$FECOSL_RUNNING" -eq 0 ] && [ ! -f "$BASE/save/fecosl_ns_running.flag" ]; then
        if has_content "$FECOSL_LS_MARKER"; then
            echo "  >>> Launching FEC-OSL NS-100..."
            touch $BASE/save/fecosl_ns_running.flag
            cd $BASE
            nohup $PYTHON -m scripts.run_fec_osl \
                --config configs/baseline_eval_ns.yml \
                --dataroot /data/datasets/nsynth-100/ \
                --dataset nsynth-100 --gpu 0 --test_times 10 \
                --out_dir save_result/end_to_end \
                > $BASE/save/run_fecosl_ns.log 2>&1 &
            echo "  FEC-OSL NS PID: $!"
        fi
    fi

    # --- Step 4: FEC-OSL FSC-89 after FEC-OSL NS completes ---
    if [ "$FECOSL_RUNNING" -eq 0 ] && [ ! -f "$BASE/save/fecosl_fsc_running.flag" ]; then
        if has_content "$BASE/save_result/end_to_end/fec_osl_nsynth-100.txt"; then
            echo "  >>> Launching FEC-OSL FSC-89..."
            touch $BASE/save/fecosl_fsc_running.flag
            cd $BASE
            nohup $PYTHON -m scripts.run_fec_osl \
                --config configs/baseline_eval_fsc89.yml \
                --dataroot /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data \
                --dataset FMC --gpu 0 --test_times 10 \
                --out_dir save_result/end_to_end \
                > $BASE/save/run_fecosl_fsc.log 2>&1 &
            echo "  FEC-OSL FSC89 PID: $!"
        fi
    fi

    # --- Step 5: Check if everything is done ---
    ALL_DONE=true
    [ "$OURS_RUNNING" -gt 0 ] && ALL_DONE=false
    [ "$EVAL_RUNNING" -gt 0 ] && ALL_DONE=false
    [ "$FECOSL_RUNNING" -gt 0 ] && ALL_DONE=false
    # Also check if Ours NS/FSC and FEC-OSL NS/FSC results exist
    [ ! -f "$BASE/save_result/opt_v4/final_exp__ns100_50x/test_result.txt" ] && [ -f "$BASE/save/ours_ns_running.flag" ] && ALL_DONE=false
    [ ! -f "$BASE/save_result/opt_v4/final_exp__fsc89_50x/test_result.txt" ] && [ -f "$BASE/save/ours_fsc_running.flag" ] && ALL_DONE=false
    [ ! -f "$BASE/save_result/end_to_end/fec_osl_nsynth-100.txt" ] && [ -f "$BASE/save/fecosl_ns_running.flag" ] && ALL_DONE=false
    [ ! -f "$BASE/save_result/end_to_end/fec_osl_FMC.txt" ] && [ -f "$BASE/save/fecosl_fsc_running.flag" ] && ALL_DONE=false

    if [ "$ALL_DONE" = true ] && [ "$OURS_RUNNING" -eq 0 ] && [ "$EVAL_RUNNING" -eq 0 ] && [ "$FECOSL_RUNNING" -eq 0 ]; then
        echo ""
        echo "============================================"
        echo "  ALL EXPERIMENTS COMPLETE!"
        echo "============================================"
        echo "  LS baseline: $(has_content $LS_EVAL_MARKER && echo 'YES' || echo 'NO')"
        echo "  NS baseline: $(has_content $NS_EVAL_MARKER && echo 'YES' || echo 'NO')"
        echo "  FSC89 baseline: $(has_content $FSC_EVAL_MARKER && echo 'YES' || echo 'NO')"
        echo "  Ours LS 50x: $(has_content $OURS_LS_MARKER && echo 'YES' || echo 'NO')"
        echo "  FEC-OSL LS: $(has_content $FECOSL_LS_MARKER && echo 'YES' || echo 'NO')"
        echo ""
        echo "  Ready to generate comparison tables."
        break
    fi
done
