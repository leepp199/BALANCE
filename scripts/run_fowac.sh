#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "$project_root"

protocol="${1:-}"
if [[ "$protocol" != "ls100" && "$protocol" != "ns100" && "$protocol" != "fsc89" ]]; then
  echo "usage: $0 {ls100|ns100|fsc89}" >&2
  exit 2
fi

python_bin="${FOWAC_PYTHON:-python}"
repeats="${FOWAC_REPEATS:-50}"
metadata_args=()
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
export TORCH_HOME="${TORCH_HOME:-${project_root}/.offline_torch}"

require_asset() {
  local value="$1"
  local label="$2"
  [[ -n "$value" ]] || { echo "$label is not set" >&2; exit 2; }
  [[ -e "$value" ]] || { echo "$label does not exist: $value" >&2; exit 2; }
}

case "$protocol" in
  ls100)
    data_root="${FOWAC_LS100_DATA:-}"
    checkpoint="${FOWAC_LS100_CHECKPOINT:-}"
    config="configs/exp_ls100.yml"
    dataset="librispeech"
    extra_args=(
      --cluster_all_candidates True
      --discovery_ranked_topk 25
      --discovery_rank_score encode_maxlogit
      --discovery_rank_start_session 3
    )
    ;;
  ns100)
    data_root="${FOWAC_NS100_DATA:-}"
    checkpoint="${FOWAC_NS100_CHECKPOINT:-}"
    config="configs/exp_ns100.yml"
    dataset="nsynth-100"
    if [[ -n "${FOWAC_NS100_METADATA:-}" ]]; then
      require_asset "$FOWAC_NS100_METADATA" FOWAC_NS100_METADATA
      metadata_args=(--ns100_metadata_root "$FOWAC_NS100_METADATA")
    fi
    extra_args=(
      --cluster_all_candidates True
      --kmeans_filter_quantile 0.5
    )
    ;;
  fsc89)
    data_root="${FOWAC_FSC89_DATA:-}"
    checkpoint="${FOWAC_FSC89_CHECKPOINT:-}"
    geometry="${FOWAC_FSC89_GEOMETRY:-}"
    metadata="${FOWAC_FSC89_METADATA:-}"
    require_asset "$geometry" FOWAC_FSC89_GEOMETRY
    require_asset "$metadata" FOWAC_FSC89_METADATA
    config="configs/exp_fsc89.yml"
    dataset="FMC"
    extra_args=(
      --cluster_all_candidates False
      --discovery_ranked_topk 25
      --discovery_rank_score encode_joint_cosine
      --base_geometry_path "$geometry"
      --fsc89_metadata_root "$metadata"
      --use_joint_cluster_assignments True
      --joint_cluster_layer layer4
      --joint_margin_weight 0.5
      --novel_bank_classifier True
      --novel_bank_topk 3
    )
    ;;
esac

require_asset "$data_root" "${protocol} data"
require_asset "$checkpoint" "${protocol} checkpoint"

structure_args=()
if [[ -n "${FOWAC_LSRB_CHECKPOINT:-}" ]]; then
  require_asset "$FOWAC_LSRB_CHECKPOINT" FOWAC_LSRB_CHECKPOINT
  structure_args=(
    --structure_discovery_checkpoint "$FOWAC_LSRB_CHECKPOINT"
    --structure_discovery_weight "${FOWAC_LSRB_WEIGHT:-0.5}"
  )
fi

"$python_bin" train_unopenset.py \
  -config "$config" \
  -dataset "$dataset" \
  --dataroot "$data_root" \
  --full_checkpoint_path "$checkpoint" \
  --skip_meta_train True \
  --mixed_openworld_stream True \
  --discovery_encoder direct \
  --encode_tta_views 1 \
  --support_proto_blend 1.0 \
  --prototype_trim_farthest 0 \
  --oracle_cluster False \
  --balanced_kmeans True \
  --balanced_kmeans_iters 5 \
  --session_restricted_alignment True \
  --reset_fc_each_round True \
  --eval_repeats "$repeats" \
  "${structure_args[@]}" \
  "${metadata_args[@]}" \
  "${extra_args[@]}"
