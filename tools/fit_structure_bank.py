#!/usr/bin/env python3
"""Fit one global MiniBatchKMeans model and save its shared structure centers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.lsrb import LatentStructureReferenceBank
from tools.lsrb_common import atomic_torch_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", default=str(ROOT / "artifacts/structure_banks/descriptors.pt"))
    parser.add_argument("--extraction-metadata", default=str(ROOT / "artifacts/structure_banks/extraction_metadata.json"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/structure_banks/structure_bank.pt"))
    parser.add_argument("--metadata", default=str(ROOT / "artifacts/structure_banks/metadata.json"))
    parser.add_argument("--num-clusters", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3420)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    payload = torch.load(cli.descriptors, map_location="cpu", weights_only=True)
    descriptors = payload["descriptors"].float()
    if descriptors.ndim != 2 or descriptors.size(1) != 512:
        raise ValueError(f"Expected descriptor matrix [N, 512], got {tuple(descriptors.shape)}")
    if descriptors.size(0) < cli.num_clusters:
        raise ValueError("The descriptor sample must contain at least K rows")

    kmeans = MiniBatchKMeans(
        n_clusters=cli.num_clusters,
        batch_size=min(cli.batch_size, descriptors.size(0)),
        max_iter=cli.max_iter,
        n_init=cli.n_init,
        random_state=cli.seed,
        reassignment_ratio=0.01,
        verbose=0,
    )
    kmeans.fit(descriptors.numpy())
    centers = torch.from_numpy(kmeans.cluster_centers_).float()
    bank = LatentStructureReferenceBank(centers, temperature=cli.temperature)
    atomic_torch_save(bank.state_dict(), cli.output)

    extraction = {}
    extraction_path = Path(cli.extraction_metadata)
    if extraction_path.exists():
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "K": bank.num_clusters,
        "feature_layer": extraction.get("feature_layer", "ResNet18.layer4.output"),
        "feature_dimension": bank.feature_dim,
        "normalization": "l2 before clustering; centers l2-normalized after fitting",
        "training_split": extraction.get("training_split", "base_train"),
        "processed_samples": extraction.get("processed_samples"),
        "descriptors_seen": extraction.get("descriptors_seen"),
        "sample_count": descriptors.size(0),
        "sampling": extraction.get("sampling", "unknown"),
        "random_seed": cli.seed,
        "algorithm": "sklearn.cluster.MiniBatchKMeans",
        "batch_size": min(cli.batch_size, descriptors.size(0)),
        "max_iter": cli.max_iter,
        "n_init": cli.n_init,
        "inertia": float(kmeans.inertia_),
        "temperature": cli.temperature,
        "source_checkpoint": extraction.get("checkpoint"),
    }
    metadata_path = Path(cli.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
