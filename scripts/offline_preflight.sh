#!/usr/bin/env bash
set -euo pipefail

cd /data/lqq/baseline_dfsb

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "OFFLINE PREFLIGHT FAILED: missing file: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "OFFLINE PREFLIGHT FAILED: missing directory: $1" >&2
    exit 2
  fi
}

require_dir /data/datasets/The_NSynth_Dataset
require_dir /data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data
require_file /data/lqq/baseline/save/exp_ns100/epoch_15.pth
require_file /data/lqq/baseline/save/exp_fsc89/epoch_15.pth
require_file configs/exp_ns100.yml
require_file configs/exp_fsc89.yml

# Fail early if the main execution path regresses to implicit URL loading.
if rg -n 'self\.encoder = resnet18\(True' network.py >/dev/null; then
  echo 'OFFLINE PREFLIGHT FAILED: MYNET still enables implicit pretrained download' >&2
  exit 3
fi

# Hugging Face/Transformers and common experiment trackers must remain offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DISABLED=true

python -m py_compile network.py train_unopenset.py threshold_free.py
python - <<'PY'
import yaml
for path in ('configs/exp_ns100.yml', 'configs/exp_fsc89.yml'):
    with open(path) as handle:
        save_dir = yaml.safe_load(handle)['train'].get('save_dir')
    if not save_dir:
        raise SystemExit(f'OFFLINE PREFLIGHT FAILED: no save_dir in {path}')
    print(f'{path}: checkpoint root={save_dir}')
PY
echo 'OFFLINE PREFLIGHT PASSED'
echo 'Required datasets, checkpoints, configs, and Python entry points are local.'
