#!/bin/bash
# Run full baseline reproduction pipeline.
#
# Runs all CIL × OSR combinations on all 3 datasets.
# Usage:
#   bash repro_baselines/run_pipeline.sh [--dry-run] [--complexity-only] [--gpu 0,1,2]
#
# GPU assignment:
#   GPU 0: LS-100 (librispeech)
#   GPU 1: NS-100 (nsynth)
#   GPU 2: FSC-89 (FMC)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

DRY_RUN=false
COMPLEXITY_ONLY=false
GPU_DEVICES="0,1,2"

# Parse args
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --complexity-only) COMPLEXITY_ONLY=true ;;
        --gpu=*) GPU_DEVICES="${arg#*=}" ;;
        --help)
            echo "Usage: $0 [--dry-run] [--complexity-only] [--gpu=0,1,2]"
            exit 0
            ;;
    esac
done

IFS=',' read -ra GPU_LIST <<< "$GPU_DEVICES"

# ============================================================
# Configuration
# ============================================================
CONFIG="$ROOT_DIR/configs/default.yml"

# CIL methods
CIL_METHODS=("pitel_cusc" "fully_fcac" "triwe" "macil" "cec" "pan" "prototypical")

# OSR methods
OSR_METHODS=("mls" "tane" "energy" "costarr" "utl" "foac_aifp" "oafn")

# Datasets
DATASETS_CONFIG=(
    "LS-100:/data/lqq/baseline/data/librispeech:80:20:100"
    "NS-100:/data/lqq/baseline/data/nsynth:80:20:100"
    "FSC-89:/data/lqq/baseline/data/FMC:69:20:89"
)

# ============================================================
# Functions
# ============================================================
run_complexity() {
    local method="$1"
    echo "========================================="
    echo "Complexity: $method"
    echo "========================================="
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] python -m repro_baselines.measure_complexity --method $method"
        return
    fi
    python -m repro_baselines.measure_complexity \
        --method "$method" \
        --num_base 80 \
        --num_all 100 2>&1 | tee -a "$ROOT_DIR/repro_baselines/logs/complexity_${method}.log"
    echo ""
}

run_train() {
    local cil="$1"
    local osr="$2"
    local dataset="$3"
    local dataroot="$4"
    local num_base="$5"
    local num_novel="$6"
    local num_all="$7"
    local gpu="$8"

    local log_dir="$ROOT_DIR/repro_baselines/logs/${dataset}/${cil}_${osr}"
    mkdir -p "$log_dir"

    echo "========================================="
    echo "Training: $cil × $osr on $dataset (GPU $gpu)"
    echo "========================================="

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] CUDA_VISIBLE_DEVICES=$gpu python -m repro_baselines.train ..."
        return
    fi

    CUDA_VISIBLE_DEVICES="$gpu" python -m repro_baselines.train \
        --config "$CONFIG" \
        --dataset "$dataset" \
        --cil "$cil" \
        --osr "$osr" \
        --num_base "$num_base" \
        --num_novel "$num_novel" \
        --num_all "$num_all" \
        --dataroot "$dataroot" \
        --seed 3420 \
        2>&1 | tee "$log_dir/train.log"

    echo ""
}

# ============================================================
# Main Pipeline
# ============================================================

# Step 1: Complexity measurement
echo "========================================="
echo "STEP 1: Complexity Measurement"
echo "========================================="

for method in "${CIL_METHODS[@]}"; do
    run_complexity "$method"
done

for method in "${OSR_METHODS[@]}"; do
    run_complexity "$method"
done

if [ "$COMPLEXITY_ONLY" = true ]; then
    echo "Complexity measurement complete."
    exit 0
fi

# Step 2: Training and evaluation
echo ""
echo "========================================="
echo "STEP 2: Training & Evaluation"
echo "========================================="

for ds_config in "${DATASETS_CONFIG[@]}"; do
    IFS=':' read -r dataset dataroot num_base num_novel num_all <<< "$ds_config"

    # Assign GPU based on dataset
    case "$dataset" in
        LS-100) GPU="${GPU_LIST[0]:-0}" ;;
        NS-100) GPU="${GPU_LIST[1]:-1}" ;;
        FSC-89) GPU="${GPU_LIST[2]:-2}" ;;
        *) GPU="0" ;;
    esac

    echo ""
    echo "--- Dataset: $dataset (GPU $GPU) ---"

    for cil in "${CIL_METHODS[@]}"; do
        for osr in "${OSR_METHODS[@]}"; do
            run_train "$cil" "$osr" "$dataset" "$dataroot" \
                      "$num_base" "$num_novel" "$num_all" "$GPU"
        done
    done
done

# Step 3: Aggregate results
echo ""
echo "========================================="
echo "STEP 3: Aggregating Results"
echo "========================================="

RESULTS_DIR="$ROOT_DIR/repro_baselines/results"
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/summary.csv"
echo "dataset,cil,osr,S0_all,S1_all,S2_all,S3_all,S4_all,AA_all,AA_inc,PD_all,auroc_s0,auroc_s4" > "$SUMMARY_FILE"

for ds_config in "${DATASETS_CONFIG[@]}"; do
    IFS=':' read -r dataset dataroot num_base num_novel num_all <<< "$ds_config"

    for cil in "${CIL_METHODS[@]}"; do
        for osr in "${OSR_METHODS[@]}"; do
            LOG_DIR="$ROOT_DIR/repro_baselines/logs/${dataset}/${cil}_${osr}"
            RESULT_FILE="$LOG_DIR/results.txt"

            if [ -f "$RESULT_FILE" ]; then
                echo "Aggregating: $dataset $cil $osr"
                # Extract key metrics
                python -c "
import re
with open('$RESULT_FILE') as fp:
    text = fp.read()

sessions = re.findall(r\"Session (\d+): \{'session': (\d+), 'all_acc': ([\d.]+), 'inc_acc': ([\d.]+), 'auroc': ([\d.]+), 'fpr95': ([\d.]+)\}\", text)

all_accs = {int(s[0]): float(s[2]) for s in sessions}
auroc = {int(s[0]): float(s[4]) for s in sessions}

line = f\"$dataset,$cil,$osr,\"
for s in range(5):
    line += f\"{all_accs.get(s, 0.0):.2f},\"
aa_all = sum(all_accs.values()) / max(len(all_accs), 1)
inc_accs = [v for k, v in all_accs.items() if k > 0]
aa_inc = sum(inc_accs) / max(len(inc_accs), 1)
pd = all_accs.get(0, 0.0) - all_accs.get(4, 0.0)
line += f\"{aa_all:.2f},{aa_inc:.2f},{pd:.2f},\"
line += f\"{auroc.get(0, 0.0):.4f},{auroc.get(4, 0.0):.4f}\"

with open('$SUMMARY_FILE', 'a') as out:
    out.write(line + '\n')
"
            else:
                echo "SKIP (no results): $dataset $cil $osr"
            fi
        done
    done
done

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "Summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
