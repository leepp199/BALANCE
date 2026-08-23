#!/bin/bash
# Monitor baseline runs and auto-update markdown
BASE=/data/lqq/baseline
PYTHON=/data/lqq/miniconda3/bin/python
NS_FLAG=$BASE/save/ns_baseline_running.flag

echo "=== Monitor started @ $(date) ==="

# Wait for a free GPU (either baseline finishes or FEC-OSL finishes)
while true; do
    sleep 300  # check every 5 minutes
    
    # Check GPU usage
    GPU0_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed -n '1p')
    GPU1_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed -n '2p')
    GPU2_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed -n '3p')
    
    # Check if FEC-OSL is still running (GPU 0)
    FECOSL=$(ps aux | grep "run_fec_osl" | grep -v grep | wc -l)
    # Check baselines
    BASELINES=$(ps aux | grep "run_all_baselines" | grep -v grep | wc -l)
    
    echo "[$(date)] GPUs: 0=${GPU0_UTIL}% 1=${GPU1_UTIL}% 2=${GPU2_UTIL}% | FEC-OSL=$FECOSL | Baselines=$BASELINES"
    
    # Start NS baseline when GPU 0 is free (FEC-OSL done) or when GPU 1/2 is free
    if [ ! -f "$NS_FLAG" ]; then
        if [ "$GPU0_UTIL" -lt 5 ] || [ "$GPU2_UTIL" -lt 5 ]; then
            # Pick a free GPU
            TARGET_GPU=""
            if [ "$GPU0_UTIL" -lt 5 ]; then
                TARGET_GPU="0"
            elif [ "$GPU2_UTIL" -lt 5 ]; then
                TARGET_GPU="2"
            fi
            
            if [ -n "$TARGET_GPU" ]; then
                echo ">>> Starting NS baseline on GPU $TARGET_GPU..."
                touch $NS_FLAG
                cd $BASE
                nohup bash -c "
                CUDA_VISIBLE_DEVICES=$TARGET_GPU $PYTHON -m scripts.run_all_baselines \
                    --config configs/baseline_eval_ns.yml \
                    --pretrained save/base_train_for_meta_ns.pth \
                    --cil cec amfo pan triwe macil \
                    --osr mls tane nci foac_aifp costarr \
                    --out_dir save_result/baselines_ns_v2 \
                    --gpu $TARGET_GPU --test_times 10 \
                    > save/run_ns_auroc.log 2>&1
                echo 'NS baseline done @ \$(date)'
                rm -f $NS_FLAG
                " > /dev/null 2>&1 &
                echo "NS baseline launched on GPU $TARGET_GPU, PID: $!"
            fi
        fi
    fi
    
    # Check if all baselines are done
    if [ "$BASELINES" -eq 0 ] && [ "$FECOSL" -eq 0 ]; then
        echo "=== ALL RUNS COMPLETE @ $(date) ==="
        break
    fi
done
