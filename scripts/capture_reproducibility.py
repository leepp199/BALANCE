#!/usr/bin/env python3
"""Write a compact, machine-readable environment/checkpoint manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


packages = {}
for name in ("torch", "torchaudio", "numpy", "scipy", "scikit-learn", "PyYAML"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = "not-installed"

tracked = [
    ROOT / "train_unopenset.py",
    ROOT / "network.py",
    ROOT / "configs/exp_ls100.yml",
    ROOT / "configs/exp_ns100.yml",
    ROOT / "configs/exp_fsc89.yml",
    Path("/data/lqq/baseline/save/backup_epoch_1777217149/epoch_15.pth"),
    Path("/data/lqq/baseline/save/exp_ns100/epoch_15.pth"),
    Path("/data/lqq/baseline/save/exp_fsc89/epoch_15.pth"),
]

manifest = {
    "captured_utc": command("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "packages": packages,
    "gpu": command(
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
    ).splitlines(),
    "files": {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in tracked
        if path.is_file()
    },
    "repeat_unit": "complete sampled open-world data stream; classifier reset each repeat",
    "formal_repeats": 10,
    "base_seed": 3420,
}

output = ROOT / "experiments/reproducibility_manifest.json"
output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(output)
