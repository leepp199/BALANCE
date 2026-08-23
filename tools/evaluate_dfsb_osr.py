#!/usr/bin/env python3
"""Training-free LS-100 OSR evaluation for the frozen DFSB.

Thresholds and score normalization are estimated only from base validation samples.
Novel test labels are used exclusively to compute metrics, never for calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--known-quantile", type=float, default=0.95)
    parser.add_argument("--semantic-weight", type=float, default=0.7)
    parser.add_argument("--structural-weight", type=float, default=0.2)
    parser.add_argument("--residual-weight", type=float, default=0.1)
    parser.add_argument("--output", default=str(ROOT / "artifacts/structure_banks/ls100_training_free_osr.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def loader_for(args, phase: str, classes: np.ndarray, batch_size: int, workers: int) -> DataLoader:
    dataset = args.Dataset.LBRS(
        root=args.dataroot, phase=phase, index=classes, k=None, base_sess=True, args=args
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def extract(model, bank, loader, device, description) -> Tuple[torch.Tensor, ...]:
    semantic, structural, residuals, labels = [], [], [], []
    original_mode = model.mode
    model.mode = "extract_feature"
    for waveforms, target in tqdm(loader, desc=description):
        feature_map = model.forward_to_layer4(waveforms.to(device, non_blocking=True))
        outputs = bank.compute(feature_map)
        semantic.append(F.normalize(F.adaptive_avg_pool2d(feature_map, 1).flatten(1), dim=-1).cpu())
        structural.append(F.normalize(outputs.structural_response, dim=-1).cpu())
        residuals.append(outputs.structure_residual.cpu())
        labels.append(target.long().cpu())
    model.mode = original_mode
    return torch.cat(semantic), torch.cat(structural), torch.cat(residuals), torch.cat(labels)


def class_means(features: torch.Tensor, labels: torch.Tensor, classes: int) -> torch.Tensor:
    means = []
    for class_id in range(classes):
        selected = features[labels == class_id]
        if selected.numel() == 0:
            raise RuntimeError(f"No base training samples for class {class_id}")
        means.append(F.normalize(selected.mean(0), dim=0))
    return torch.stack(means)


def component_distances(z, h, residual, semantic_prototypes, structural_prototypes):
    semantic_per_class = 1.0 - z @ semantic_prototypes.t()
    structural_per_class = 1.0 - h @ structural_prototypes.t()
    return semantic_per_class, structural_per_class, residual


def robust_scale_from_known(values: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    median = float(values.median())
    q25 = float(torch.quantile(values, 0.25))
    q75 = float(torch.quantile(values, 0.75))
    scale = max(q75 - q25, 1e-6)
    return (values - median) / scale, median, scale


def apply_scale(values: torch.Tensor, median: float, scale: float) -> torch.Tensor:
    return (values - median) / scale


def metrics(known_score, unknown_score, quantile):
    threshold = float(torch.quantile(known_score, quantile))
    scores = torch.cat((known_score, unknown_score)).numpy()
    targets = np.concatenate((np.zeros(len(known_score)), np.ones(len(unknown_score))))
    fpr, tpr, thresholds = roc_curve(targets, scores)
    valid = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[valid[0]]) if valid.size else 1.0
    return {
        "auroc": float(roc_auc_score(targets, scores)),
        "aupr_unknown": float(average_precision_score(targets, scores)),
        "fpr95": fpr95,
        "threshold_base_val_q": threshold,
        "known_acceptance": float((known_score <= threshold).float().mean()),
        "unknown_recall": float((unknown_score > threshold).float().mean()),
        "balanced_osr_accuracy": float(0.5 * ((known_score <= threshold).float().mean()
                                               + (unknown_score > threshold).float().mean())),
    }


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    args = load_project_args(cli.config, cli.dataset, cli.dataroot)
    from train_unopenset import set_up_datasets
    set_up_datasets(args)
    model, _, _ = build_model(args, cli.checkpoint, device)
    bank = DeepFeatureStructureBank.load(cli.bank).to(device)

    _, base_train_loader = build_base_loader(args, cli.batch_size, cli.workers)
    val_loader = loader_for(args, "val", np.arange(args.num_base), cli.batch_size, cli.workers)
    known_loader = loader_for(args, "test", np.arange(args.num_base), cli.batch_size, cli.workers)
    unknown_loader = loader_for(args, "test", np.arange(args.num_base, args.num_all), cli.batch_size, cli.workers)

    z_train, h_train, _, y_train = extract(model, bank, base_train_loader, device, "Base prototypes")
    p_sem = class_means(z_train, y_train, args.num_base)
    p_str = class_means(h_train, y_train, args.num_base)
    del z_train, h_train, y_train

    z_val, h_val, e_val, _ = extract(model, bank, val_loader, device, "Base validation calibration")
    z_known, h_known, e_known, _ = extract(model, bank, known_loader, device, "Known test")
    z_unknown, h_unknown, e_unknown, _ = extract(model, bank, unknown_loader, device, "Unknown test")

    val_sem_pc, val_str_pc, _ = component_distances(z_val, h_val, e_val, p_sem, p_str)
    known_sem_pc, known_str_pc, _ = component_distances(z_known, h_known, e_known, p_sem, p_str)
    unknown_sem_pc, unknown_str_pc, _ = component_distances(z_unknown, h_unknown, e_unknown, p_sem, p_str)

    # Score each sample against a coherent class pair, not independent nearest classes.
    val_sem = val_sem_pc.min(1).values
    known_sem = known_sem_pc.min(1).values
    unknown_sem = unknown_sem_pc.min(1).values
    _, sem_med, sem_scale = robust_scale_from_known(val_sem)
    _, str_med, str_scale = robust_scale_from_known(val_str_pc.min(1).values)
    _, res_med, res_scale = robust_scale_from_known(e_val)

    def fused(sem_pc, str_pc, residual, include_residual):
        sem = apply_scale(sem_pc, sem_med, sem_scale)
        structure = apply_scale(str_pc, str_med, str_scale)
        paired = (cli.semantic_weight * sem + cli.structural_weight * structure).min(1).values
        if include_residual:
            paired = paired + cli.residual_weight * apply_scale(residual, res_med, res_scale)
        return paired

    scores = {
        "semantic_only": (val_sem, known_sem, unknown_sem),
        "semantic_structural": (
            fused(val_sem_pc, val_str_pc, e_val, False),
            fused(known_sem_pc, known_str_pc, e_known, False),
            fused(unknown_sem_pc, unknown_str_pc, e_unknown, False),
        ),
        "semantic_structural_residual": (
            fused(val_sem_pc, val_str_pc, e_val, True),
            fused(known_sem_pc, known_str_pc, e_known, True),
            fused(unknown_sem_pc, unknown_str_pc, e_unknown, True),
        ),
    }
    report: Dict[str, object] = {
        "protocol": "training-free; base-validation-only calibration",
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "bank": str(Path(cli.bank).resolve()),
        "K": bank.num_clusters,
        "base_classes": args.num_base,
        "novel_classes": args.num_all - args.num_base,
        "known_quantile": cli.known_quantile,
        "weights": {"semantic": cli.semantic_weight, "structural": cli.structural_weight,
                    "residual": cli.residual_weight},
        "counts": {"base_val": len(z_val), "known_test": len(z_known), "unknown_test": len(z_unknown)},
        "results": {},
    }
    for name, (validation_score, known_score, unknown_score) in scores.items():
        result = metrics(known_score, unknown_score, cli.known_quantile)
        result["threshold_base_val_q"] = float(torch.quantile(validation_score, cli.known_quantile))
        threshold = result["threshold_base_val_q"]
        result["known_acceptance"] = float((known_score <= threshold).float().mean())
        result["unknown_recall"] = float((unknown_score > threshold).float().mean())
        result["balanced_osr_accuracy"] = 0.5 * (result["known_acceptance"] + result["unknown_recall"])
        report["results"][name] = result

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
