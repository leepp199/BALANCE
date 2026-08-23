#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "$project_root"

python_bin="${FOWAC_PYTHON:-python}"
require_assets="${FOWAC_REQUIRE_ASSETS:-0}"

fail() {
  echo "OFFLINE PREFLIGHT FAILED: $*" >&2
  exit 2
}

require_file() {
  [[ -s "$1" ]] || fail "missing or empty file: $1"
}

check_optional_dir() {
  local variable_name="$1"
  local label="$2"
  local value="${!variable_name:-}"
  if [[ -n "$value" ]]; then
    [[ -d "$value" ]] || fail "$label directory does not exist: $value"
    echo "asset ok: $label=$value"
  elif [[ "$require_assets" == "1" ]]; then
    fail "$variable_name is required when FOWAC_REQUIRE_ASSETS=1"
  else
    echo "asset skipped: $variable_name is not set"
  fi
}

check_optional_file() {
  local variable_name="$1"
  local label="$2"
  local required_when_strict="${3:-1}"
  local value="${!variable_name:-}"
  if [[ -n "$value" ]]; then
    [[ -s "$value" ]] || fail "$label file does not exist or is empty: $value"
    echo "asset ok: $label=$value"
  elif [[ "$require_assets" == "1" && "$required_when_strict" == "1" ]]; then
    fail "$variable_name is required when FOWAC_REQUIRE_ASSETS=1"
  else
    echo "asset skipped: $variable_name is not set"
  fi
}

# Prevent libraries and experiment trackers from initiating network access.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
export TORCH_HOME="${TORCH_HOME:-${project_root}/.offline_torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/fowac-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

core_files=(
  requirements.txt
  network.py
  threshold_free.py
  train_unopenset.py
  models/AttnClassifier.py
  models/FSEval.py
  models/metatrainer_oo.py
  models/lsrb.py
  models/resnet18_encoder.py
  models/uncertainty.py
  scripts/run_fowac.sh
  configs/exp_ls100.yml
  configs/exp_ns100.yml
  configs/exp_fsc89.yml
)
for path in "${core_files[@]}"; do
  require_file "$path"
done

bash -n scripts/run_fowac.sh

# Fail if the primary network constructor is changed back to implicit downloads.
if command -v rg >/dev/null 2>&1; then
  if rg -n 'self\.encoder = resnet18\(True' network.py >/dev/null; then
    fail "network.py enables implicit pretrained-weight downloading"
  fi
else
  if grep -n 'self\.encoder = resnet18(True' network.py >/dev/null; then
    fail "network.py enables implicit pretrained-weight downloading"
  fi
fi

"$python_bin" -m py_compile \
  network.py \
  threshold_free.py \
  train_unopenset.py \
  models/AttnClassifier.py \
  models/FSEval.py \
  models/metatrainer_oo.py \
  models/lsrb.py \
  models/resnet18_encoder.py \
  models/uncertainty.py

"$python_bin" - <<'PY'
import importlib

import yaml

configs = {
    "configs/exp_ls100.yml": (80, 100),
    "configs/exp_ns100.yml": (80, 100),
    "configs/exp_fsc89.yml": (69, 89),
}
for path, (expected_base, expected_all) in configs.items():
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("train"), dict):
        raise SystemExit(f"OFFLINE PREFLIGHT FAILED: invalid train config: {path}")
    train = payload["train"]
    if train.get("num_base") != expected_base or train.get("num_all") != expected_all:
        raise SystemExit(
            f"OFFLINE PREFLIGHT FAILED: protocol mismatch in {path}: "
            f"num_base={train.get('num_base')} num_all={train.get('num_all')}"
        )
    if train.get("num_session") != 5 or train.get("way") != 5 or train.get("shot") != 5:
        raise SystemExit(f"OFFLINE PREFLIGHT FAILED: session protocol mismatch in {path}")
    print(f"config ok: {path}")

modules = (
    "models.AttnClassifier",
    "models.FSEval",
    "models.metatrainer_oo",
    "models.lsrb",
    "models.resnet18_encoder",
    "models.uncertainty",
    "network",
    "threshold_free",
    "train_unopenset",
    "data.librispeech",
    "data.nsynth",
    "data.FMC",
)
for name in modules:
    importlib.import_module(name)
    print(f"import ok: {name}")
PY

check_optional_dir FOWAC_LS100_DATA "LS-100 data"
check_optional_dir FOWAC_NS100_DATA "NS-100 data"
check_optional_dir FOWAC_FSC89_DATA "FSC-89 data"
check_optional_dir FOWAC_NS100_METADATA "NS-100 metadata"
check_optional_dir FOWAC_FSC89_METADATA "FSC-89 metadata"
check_optional_file FOWAC_LS100_CHECKPOINT "LS-100 checkpoint"
check_optional_file FOWAC_NS100_CHECKPOINT "NS-100 checkpoint"
check_optional_file FOWAC_FSC89_CHECKPOINT "FSC-89 checkpoint"
check_optional_file FOWAC_LSRB_CHECKPOINT "LSRB checkpoint"
check_optional_file FOWAC_FSC89_GEOMETRY "FSC-89 geometry"

echo "OFFLINE PREFLIGHT PASSED"
echo "Source, protocol configurations, and core imports are available offline."
