#!/usr/bin/env python3
"""Validate shared-bank assignment, response, and residual on multiple samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dfsb import DeepFeatureStructureBank
from tools.dfsb_common import build_base_loader, build_model, load_project_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/exp_ls100.yml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", default=str(ROOT / "artifacts/structure_banks/structure_bank.pt"))
    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--dataroot", default="/data/datasets/librispeech_fscil/")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    project_args = load_project_args(cli.config, cli.dataset, cli.dataroot)
    _, loader = build_base_loader(project_args, cli.batch_size, cli.workers)
    model, _, _ = build_model(project_args, cli.checkpoint, device)
    bank = DeepFeatureStructureBank.load(cli.bank).to(device)

    batch = next(iter(loader))
    waveforms = batch[0].to(device)
    with torch.inference_mode():
        feature_map = model.forward_to_layer4(waveforms)
        outputs = bank.compute(feature_map)
    response_sums = outputs.structural_response.sum(dim=-1)
    report = {
        "shared_bank_object_id": id(bank),
        "bank_shape": list(bank.centers.shape),
        "feature_map_shape": list(feature_map.shape),
        "assignments_shape": list(outputs.assignments.shape),
        "structural_response_shape": list(outputs.structural_response.shape),
        "structure_residual_shape": list(outputs.structure_residual.shape),
        "assignment_min": int(outputs.assignments.min()),
        "assignment_max": int(outputs.assignments.max()),
        "response_sum_min": float(response_sums.min()),
        "response_sum_max": float(response_sums.max()),
        "residual_mean": float(outputs.structure_residual.mean()),
        "residual_std": float(outputs.structure_residual.std(unbiased=False)),
        "all_finite": bool(
            torch.isfinite(outputs.structural_response).all()
            and torch.isfinite(outputs.structure_residual).all()
        ),
        "same_bank_for_all_samples": True,
    }
    if not report["all_finite"] or not torch.allclose(response_sums, torch.ones_like(response_sums), atol=1e-5):
        raise RuntimeError(f"DFSB validation failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
