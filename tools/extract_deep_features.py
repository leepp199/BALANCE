#!/usr/bin/env python3
"""Extract a uniform bounded sample of layer4 descriptors from the base train split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dfsb import descriptor_matrix
from tools.dfsb_common import atomic_torch_save, build_base_loader, build_model, load_project_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/exp_ls100.yml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--dataroot", default="/data/datasets/librispeech_fscil/")
    parser.add_argument("--output", default=str(ROOT / "artifacts/structure_banks/descriptors.pt"))
    parser.add_argument("--metadata", default=str(ROOT / "artifacts/structure_banks/extraction_metadata.json"))
    parser.add_argument("--max-descriptors", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means the full base split")
    parser.add_argument("--seed", type=int, default=3420)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    device = torch.device(cli.device)
    project_args = load_project_args(cli.config, cli.dataset, cli.dataroot)
    project_args.seed = cli.seed
    dataset, loader = build_base_loader(project_args, cli.batch_size, cli.workers)
    model, missing, unexpected = build_model(project_args, cli.checkpoint, device)

    key_generator = torch.Generator(device="cpu").manual_seed(cli.seed)
    kept_descriptors = torch.empty((0, 512), dtype=torch.float32)
    kept_keys = torch.empty((0,), dtype=torch.float32)
    total_descriptors = 0
    sample_count = 0
    observed_shape = None

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="Extract layer4 descriptors")):
            if cli.max_batches and batch_index >= cli.max_batches:
                break
            waveforms = batch[0].to(device, non_blocking=True)
            feature_map = model.forward_to_layer4(waveforms, augment=False)
            observed_shape = list(feature_map.shape)
            descriptors = descriptor_matrix(feature_map)
            keys = torch.rand(descriptors.size(0), generator=key_generator)
            total_descriptors += descriptors.size(0)
            sample_count += feature_map.size(0)

            candidates = torch.cat((kept_descriptors, descriptors), dim=0)
            candidate_keys = torch.cat((kept_keys, keys), dim=0)
            keep = min(cli.max_descriptors, candidate_keys.numel())
            indices = torch.topk(candidate_keys, k=keep, sorted=False).indices
            kept_descriptors = candidates[indices]
            kept_keys = candidate_keys[indices]

    if kept_descriptors.numel() == 0:
        raise RuntimeError("No descriptors were extracted")
    atomic_torch_save(
        {"descriptors": kept_descriptors, "seed": cli.seed, "normalization": "l2"},
        cli.output,
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": cli.dataset,
        "training_split": "base_train",
        "dataset_samples": len(dataset),
        "processed_samples": sample_count,
        "feature_layer": "ResNet18.layer4.output",
        "observed_batch_feature_shape": observed_shape,
        "feature_dimension": kept_descriptors.size(1),
        "normalization": "l2",
        "sampling": "bounded uniform random-key reservoir",
        "descriptors_seen": total_descriptors,
        "descriptors_kept": kept_descriptors.size(0),
        "random_seed": cli.seed,
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_missing_noncritical_keys": list(missing),
        "checkpoint_unexpected_keys": list(unexpected),
    }
    metadata_path = Path(cli.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
