#!/usr/bin/env python3
"""Export DFSB pseudo targets and sample-level structural statistics for inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dfsb import DeepFeatureStructureBank
from tools.dfsb_common import atomic_torch_save, build_base_loader, build_model, load_project_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/exp_ls100.yml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", default=str(ROOT / "artifacts/structure_banks/structure_bank.pt"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/structure_banks/structure_targets_preview.pt"))
    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--dataroot", default="/data/datasets/librispeech_fscil/")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()

    device = torch.device(cli.device)
    args = load_project_args(cli.config, cli.dataset, cli.dataroot)
    _, loader = build_base_loader(args, cli.batch_size, cli.workers)
    model, _, _ = build_model(args, cli.checkpoint, device)
    bank = DeepFeatureStructureBank.load(cli.bank).to(device)
    records = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= cli.max_batches:
                break
            feature_map = model.forward_to_layer4(batch[0].to(device))
            outputs = bank.compute(feature_map)
            records.append({
                "labels": batch[1].cpu(),
                "assignments": outputs.assignments.cpu(),
                "structural_response": outputs.structural_response.cpu(),
                "structure_residual": outputs.structure_residual.cpu(),
            })
    atomic_torch_save({"bank_path": str(Path(cli.bank).resolve()), "batches": records}, cli.output)
    print(f"saved {len(records)} batches to {cli.output}")


if __name__ == "__main__":
    main()
