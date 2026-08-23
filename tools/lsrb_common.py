"""Shared CLI helpers for offline LSRB construction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return argparse.Namespace(**{k: _namespace(v) for k, v in value.items()})
    return value


def load_project_args(config_path: str, dataset: str, dataroot: str) -> argparse.Namespace:
    """Load the existing train YAML and add the CLI fields MYNET expects."""
    # Reuse all defaults from the real training entry so auxiliary classifier modules
    # can be constructed even though Phase B only consumes the encoder.
    from train_unopenset import args_parser

    with open(config_path, "r", encoding="utf-8") as handle:
        config: Dict[str, Any] = yaml.safe_load(handle)["train"]
    merged: Dict[str, Any] = vars(args_parser().parse_args([]))
    merged.update(config)
    merged.update({
        "dataset": dataset,
        "dataroot": dataroot,
        "train_weight_base": 1,
        "seed": int(config.get("seed", 42)),
        "cuda": torch.cuda.is_available(),
        "num_labeled_classes": int(config.get("num_base", 80)),
    })
    return _namespace(merged)


def build_model(args: argparse.Namespace, checkpoint: str, device: torch.device):
    from network import MYNET

    model = MYNET(args, mode="extract_feature").to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("params", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Classifier-side differences are tolerated; encoder/bn0 mismatches are not.
    critical_missing = [k for k in missing if k.startswith(("encoder.", "bn0."))]
    if critical_missing:
        raise RuntimeError(f"Checkpoint is missing feature-extractor keys: {critical_missing[:10]}")
    model.eval()
    return model, missing, unexpected


def build_base_loader(args: argparse.Namespace, batch_size: int, workers: int):
    from train_unopenset import set_up_datasets
    from data.dataloader import get_pretrain_dataloader

    set_up_datasets(args)
    dataset, _ = get_pretrain_dataloader(args)
    generator = torch.Generator().manual_seed(int(args.seed))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    return dataset, loader


def atomic_torch_save(payload: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
