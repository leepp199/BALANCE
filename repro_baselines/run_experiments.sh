#!/bin/bash
# Run Group 1 + Group 2 baseline experiments.
# As per baseline_compare.md Section 9.
#
# GPU usage: LS-100 on GPU 0
# Usage: bash repro_baselines/run_experiments.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Not using set -e: one experiment failure should not abort others

# ============================================================
# Experiment Configuration
# ============================================================

# Group 1: Classic sanity baselines
GROUP1_CIL=("prototypical" "cec" "cec" "pan")
GROUP1_OSR=("mls" "mls" "tane" "mls")

# Group 2: Strong 2025-2026 combined baselines
GROUP2_CIL=("pitel_cusc" "pitel_cusc" "fully_fcac" "fully_fcac" "triwe" "triwe" "macil" "macil")
GROUP2_OSR=("costarr" "foac_aifp" "costarr" "foac_aifp" "costarr" "utl" "costarr" "foac_aifp")

# All experiments
ALL_CIL=("${GROUP1_CIL[@]}" "${GROUP2_CIL[@]}")
ALL_OSR=("${GROUP1_OSR[@]}" "${GROUP2_OSR[@]}")

# Datasets
DATASETS=("librispeech")
DATAROOTS=("/data/datasets/librispeech_fscil/")
NUM_BASES=(80)
NUM_NOVELS=(20)
NUM_ALLS=(100)
GPUS=(0)

# Override for multiple datasets
# DATASETS=("librispeech" "nsynth" "FMC")
# DATAROOTS=("/data/datasets/librispeech_fscil/" "/data/datasets/The_NSynth_Dataset/nsynth-100-fs-meta/" "/data/datasets/FSD-MIX-CLIPS-for_FSCIL/")
# NUM_BASES=(80 80 69)
# NUM_NOVELS=(20 20 20)
# NUM_ALLS=(100 100 89)
# GPUS=(0 1 2)

mkdir -p "$ROOT_DIR/repro_baselines/logs"
mkdir -p "$ROOT_DIR/repro_baselines/results"

echo "========================================="
echo "Starting baseline comparison experiments"
echo "Total combos: ${#ALL_CIL[@]}"
echo "Datasets: ${DATASETS[*]}"
echo "========================================="
echo ""

for didx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$didx]}"
    dataroot="${DATAROOTS[$didx]}"
    num_base="${NUM_BASES[$didx]}"
    num_novel="${NUM_NOVELS[$didx]}"
    num_all="${NUM_ALLS[$didx]}"
    gpu="${GPUS[$didx]}"

    echo "--- Dataset: $dataset (GPU $gpu) ---"

    for eidx in "${!ALL_CIL[@]}"; do
        cil="${ALL_CIL[$eidx]}"
        osr="${ALL_OSR[$eidx]}"
        
        LOG_DIR="$ROOT_DIR/repro_baselines/logs/${dataset}/${cil}_${osr}"
        RESULT_FILE="$LOG_DIR/results.txt"
        
        if [ -f "$RESULT_FILE" ]; then
            echo "  SKIP (already done): $cil × $osr"
            continue
        fi

        echo "  Running: $cil × $osr (GPU $gpu)"
        
        mkdir -p "$LOG_DIR"
        
# GPU already selected via CUDA_VISIBLE_DEVICES
        CUDA_VISIBLE_DEVICES="$gpu" python -m repro_baselines.train \
            --cil "$cil" \
            --osr "$osr" \
            --dataset "$dataset" \
            --dataroot "$dataroot" \
            --num_base "$num_base" \
            --num_novel "$num_novel" \
            --num_all "$num_all" \
            --gpu 0 \
            > "$LOG_DIR/train.log" 2>&1

        # Check exit status
        if [ $? -eq 0 ]; then
            echo "  Done: $cil × $osr"
        else
            echo "  FAILED: $cil × $osr (check $LOG_DIR/train.log)"
        fi
        echo ""
    done
done

# ============================================================
# Aggregate results
# ============================================================
echo ""
echo "========================================="
echo "Aggregating results"
echo "========================================="

SUMMARY_FILE="$ROOT_DIR/repro_baselines/results/summary.csv"
echo "dataset,cil,osr,S0_all,S1_all,S2_all,S3_all,S4_all,AA_all,AA_inc,PD_all,AUROC_S0,AUROC_S4,FPR95_S0,FPR95_S4" > "$SUMMARY_FILE"

for didx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$didx]}"
    for eidx in "${!ALL_CIL[@]}"; do
        cil="${ALL_CIL[$eidx]}"
        osr="${ALL_OSR[$eidx]}"
        
        RESULT_FILE="$ROOT_DIR/repro_baselines/logs/${dataset}/${cil}_${osr}/results.txt"
        
        if [ ! -f "$RESULT_FILE" ]; then
            echo "WARNING: Missing $RESULT_FILE"
            continue
        fi

        # Parse results
        line="$dataset,$cil,$osr"
        for s in 0 1 2 3 4; do
            acc=$(grep "Session $s:" "$RESULT_FILE" | grep -oP "'all_acc': [\d.]+" | grep -oP "[\d.]+")
            if [ -z "$acc" ]; then acc="0.0"; fi
            line="$line,$acc"
        done
        
        # AA_all
        aa_all=$(grep "^AA_all:" "$RESULT_FILE" | grep -oP "[\d.]+")
        if [ -z "$aa_all" ]; then aa_all="0.0"; fi
        line="$line,$aa_all"
        
        # AA_inc
        aa_inc=$(grep "^AA_inc:" "$RESULT_FILE" | grep -oP "[\d.]+")
        if [ -z "$aa_inc" ]; then aa_inc="0.0"; fi
        line="$line,$aa_inc"
        
        # PD_all
        pd=$(grep "^PD_all:" "$RESULT_FILE" | grep -oP "[\d.]+")
        if [ -z "$pd" ]; then pd="0.0"; fi
        line="$line,$pd"
        
        # AUROC
        for s in 0 4; do
            ar=$(grep "Session $s:" "$RESULT_FILE" | grep -oP "'auroc': [\d.]+" | grep -oP "[\d.]+")
            if [ -z "$ar" ]; then ar="0.0"; fi
            line="$line,$ar"
        done
        
        # FPR95
        for s in 0 4; do
            fp=$(grep "Session $s:" "$RESULT_FILE" | grep -oP "'fpr95': [\d.]+" | grep -oP "[\d.]+")
            if [ -z "$fp" ]; then fp="0.0"; fi
            line="$line,$fp"
        done
        
        echo "$line" >> "$SUMMARY_FILE"
    done
done

echo ""
echo "Summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
echo ""
echo "Done!"
