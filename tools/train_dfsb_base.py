#!/usr/bin/env python3
"""Phase C: structure-guided base training from a supervised warm-up checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dfsb import DeepFeatureStructureBank, MaskedStructurePredictor
from tools.dfsb_common import atomic_torch_save, build_base_loader, build_model, load_project_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/exp_ls100.yml"))
    parser.add_argument("--warmup-checkpoint", required=True)
    parser.add_argument("--bank", default=str(ROOT / "artifacts/structure_banks/structure_bank.pt"))
    parser.add_argument("--output", default=str(ROOT / "artifacts/checkpoints/dfsb_ls100_base.pth"))
    parser.add_argument("--history", default=str(ROOT / "artifacts/checkpoints/dfsb_ls100_history.json"))
    parser.add_argument("--dataset", default="librispeech")
    parser.add_argument("--dataroot", default="/data/datasets/librispeech_fscil/")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--structure-weight", type=float, default=0.1)
    parser.add_argument("--feature-anchor-weight", type=float, default=0.0,
                        help="MSE weight to the frozen warm-up layer4 representation")
    parser.add_argument("--logit-anchor-weight", type=float, default=0.0,
                        help="KL weight to the frozen warm-up classifier distribution")
    parser.add_argument("--anchor-temperature", type=float, default=2.0)
    parser.add_argument("--train-scope", choices=["all", "layer4", "predictor"], default="all",
                        help="Encoder scope updated by structure-guided training")
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--mask-mode", choices=["random", "axis_h", "axis_w", "dual_axis"], default="random")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3420)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def evaluate_base(model, loader, device):
    model.eval()
    correct = total = 0
    original_mode = model.mode
    model.mode = "extract_feature"
    for waveforms, labels in loader:
        fmap = model.forward_to_layer4(waveforms.to(device, non_blocking=True))
        features = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        predictions = model.fc(features).argmax(1).cpu()
        correct += int((predictions == labels).sum())
        total += labels.numel()
    model.mode = original_mode
    return correct / max(total, 1)


def main():
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    device = torch.device(cli.device)
    args = load_project_args(cli.config, cli.dataset, cli.dataroot)
    args.seed = cli.seed
    dataset, _ = build_base_loader(args, cli.batch_size, cli.workers)
    train_generator = torch.Generator().manual_seed(cli.seed)
    loader = DataLoader(
        dataset, batch_size=cli.batch_size, shuffle=True, num_workers=cli.workers,
        pin_memory=torch.cuda.is_available(), generator=train_generator,
    )
    model, missing, unexpected = build_model(args, cli.warmup_checkpoint, device)
    teacher, _, _ = build_model(args, cli.warmup_checkpoint, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    bank = DeepFeatureStructureBank.load(cli.bank).to(device)
    predictor = MaskedStructurePredictor(
        feature_dim=bank.feature_dim, num_clusters=bank.num_clusters,
        hidden_dim=cli.hidden_dim, mask_ratio=cli.mask_ratio, mask_mode=cli.mask_mode,
    ).to(device)

    if cli.train_scope != "all":
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in model.bn0.parameters():
            parameter.requires_grad_(False)
    if cli.train_scope == "layer4":
        for parameter in model.encoder.layer4.parameters():
            parameter.requires_grad_(True)
    if cli.train_scope == "predictor":
        for parameter in model.fc.parameters():
            parameter.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters += list(predictor.parameters())
    optimizer = torch.optim.SGD(parameters, lr=cli.lr, momentum=0.9, nesterov=True,
                                weight_decay=cli.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli.epochs)
    history = []
    best_loss = float("inf")
    output_path = Path(cli.output)

    for epoch in range(cli.epochs):
        model.train()
        predictor.train()
        model.mode = "extract_feature"
        sums = {"total": 0.0, "cls": 0.0, "str": 0.0, "feat_anchor": 0.0,
                "logit_anchor": 0.0, "str_acc": 0.0, "acc": 0.0}
        count = 0
        progress = tqdm(loader, desc=f"DFSB epoch {epoch + 1}/{cli.epochs}")
        for waveforms, labels in progress:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feature_map = model.forward_to_layer4(waveforms, augment=False)
            features = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
            logits = model.fc(features)
            loss_cls = F.cross_entropy(logits, labels)
            loss_str, str_acc, _, _ = predictor(feature_map, bank.centers)
            with torch.no_grad():
                teacher_map = teacher.forward_to_layer4(waveforms, augment=False)
                teacher_features = F.adaptive_avg_pool2d(teacher_map, 1).flatten(1)
                teacher_logits = teacher.fc(teacher_features)
            loss_feat_anchor = F.mse_loss(
                F.normalize(feature_map, dim=1), F.normalize(teacher_map, dim=1))
            temperature = cli.anchor_temperature
            loss_logit_anchor = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1), reduction="batchmean") * temperature**2
            loss = (loss_cls + cli.structure_weight * loss_str
                    + cli.feature_anchor_weight * loss_feat_anchor
                    + cli.logit_anchor_weight * loss_logit_anchor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch = labels.numel()
            sums["total"] += float(loss.detach()) * batch
            sums["cls"] += float(loss_cls.detach()) * batch
            sums["str"] += float(loss_str.detach()) * batch
            sums["feat_anchor"] += float(loss_feat_anchor.detach()) * batch
            sums["logit_anchor"] += float(loss_logit_anchor.detach()) * batch
            sums["str_acc"] += float(str_acc) * batch
            sums["acc"] += float((logits.argmax(1) == labels).float().sum())
            count += batch
            progress.set_postfix(loss=f"{sums['total']/count:.3f}",
                                 acc=f"{sums['acc']/count:.3f}", str_acc=f"{sums['str_acc']/count:.3f}")
        scheduler.step()
        record = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                  **{key: value / count for key, value in sums.items()}}
        history.append(record)
        payload = {
            "params": model.state_dict(),
            "structure_predictor": predictor.state_dict(),
            "structure_bank": bank.state_dict(),
            "training": vars(cli),
            "history": history,
        }
        atomic_torch_save(payload, output_path.with_name(f"{output_path.stem}_epoch{epoch+1}.pth"))
        if record["total"] < best_loss:
            best_loss = record["total"]
            atomic_torch_save(payload, output_path)
        Path(cli.history).parent.mkdir(parents=True, exist_ok=True)
        Path(cli.history).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record))
    print(f"saved best DFSB checkpoint to {output_path}")


if __name__ == "__main__":
    main()
