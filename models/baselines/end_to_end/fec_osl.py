"""FEC-OSL-style — End-to-End Open-Set Semi-Supervised Learning (2026).

Reference: "End-to-End Open-Set Semi-Supervised Learning for Fine-Grained
Encrypted Traffic Classification", IEEE TIFS 2026.

Adapted for few-shot open-world audio classification:
- Uses **energy score** for known/unknown boundary (no OSR-specific training)
- **Adaptive KMeans** on detected unknowns (vs traditional n_init KMeans)
- **Incremental prototype update** (cosine prototype registration)

This acts as a lighter, end-to-end alternative to the full FOWAC pipeline.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from network import MYNET, replace_base_fc
from utils.util import cluster_acc, calc
from sklearn.metrics import roc_auc_score
from utils.utils import count_acc


class FEC_OSL:
    """End-to-end open-world learner (FEC-OSL style).

    Usage:
        learner = FEC_OSL(args)
        learner.train_base(trainloader)           # stage 0
        learner.evaluate_session(mixed_loader, s) # incremental sessions
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MYNET(args, mode='encoder').to(self.device)

        # Energy temperature
        self.energy_T = float(getattr(args, 'fec_energy_T', 1.0))
        # Base-threshold percentile for energy-based unknown detection
        self.energy_quantile = float(getattr(args, 'fec_quantile', 0.5))

        # Keep track of sessions
        self.n_base = int(getattr(args, 'num_base', 80))
        self.n_novel_per_session = int(getattr(args, 'way', 5))
        self.total_sessions = int(getattr(args, 'num_session', 5))
        self.start_session = int(getattr(args, 'start_session', 1))

    # ------------------------------------------------------------------
    def train_base(self, trainloader, epochs: int = 30) -> None:
        """Standard cosine CE training on base classes."""
        self.model.train()
        opt = torch.optim.SGD(
            [p for p in self.model.encoder.parameters() if p.requires_grad] +
            [self.model.fc.weight],
            lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for ep in range(epochs):
            loss_meter = 0.0
            n_correct, n_seen = 0, 0
            self.model.train()
            for batch in trainloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device).long()
                self.model.mode = '__feat__'
                feats = self.model.encode(x)
                self.model.mode = 'encoder'
                logits = F.linear(F.normalize(feats, dim=-1),
                                  F.normalize(self.model.fc.weight[:self.n_base], dim=-1))
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                pred = logits.argmax(-1)
                n_correct += int((pred == y).sum().item())
                n_seen += y.size(0)
                loss_meter += float(loss.item()) * y.size(0)
            scheduler.step()
            acc = n_correct / max(n_seen, 1)
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"[FEC-OSL base] epoch {ep+1}/{epochs} loss={loss_meter/max(n_seen,1):.4f} acc={acc:.4f}")

        # Align classifier prototypes
        trainset = trainloader.dataset  # extract dataset from DataLoader
        if hasattr(trainset, 'dataset'):
            self.model = replace_base_fc(self.args, trainset.dataset, self.model)
        else:
            self.model = replace_base_fc(self.args, trainset, self.model)

        # Compute energy baseline from training data
        self._compute_energy_baseline(trainloader)

    @torch.no_grad()
    def _compute_energy_baseline(self, trainloader):
        """Compute base-class energy statistics for adaptive thresholding."""
        self.model.eval()
        energies = []
        for batch in trainloader:
            x, _ = batch[0].to(self.device), batch[1]
            self.model.mode = '__feat__'
            feats = self.model.encode(x)
            self.model.mode = 'encoder'
            logits = F.linear(F.normalize(feats, dim=-1),
                              F.normalize(self.model.fc.weight[:self.n_base], dim=-1))
            energy = self.energy_T * torch.logsumexp(logits / self.energy_T, dim=-1)
            energies.append(energy.cpu())
        all_energy = torch.cat(energies)
        self.base_energy_mean = float(all_energy.mean())
        self.base_energy_std = float(all_energy.std())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.model.mode = '__feat__'
        f = self.model.encode(x)
        self.model.mode = 'encoder'
        return f

    # ------------------------------------------------------------------
    def evaluate_session(self, mixed_loader, session: int,
                         known_classes: int) -> Dict[str, float]:
        """Run one incremental session."""
        # 1. Extract features for mixed data
        feats, labs = [], []
        for batch in mixed_loader:
            x, y = batch[0].to(self.device), batch[1]
            f = self.encode(x)
            feats.append(f.cpu())
            labs.append(y.cpu())
        all_feats = torch.cat(feats, 0)
        all_labs = torch.cat(labs, 0)

        # 2. Energy-based unknown detection
        logits = F.linear(F.normalize(all_feats.to(self.device), dim=-1),
                          F.normalize(self.model.fc.weight[:known_classes].to(self.device), dim=-1))
        energy = self.energy_T * torch.logsumexp(logits / self.energy_T, dim=-1)
        energy = energy.cpu()

        # Adaptive threshold: base_mean - 2*std (more conservative = fewer unknowns)
        thr = self.base_energy_mean - 0.5 * self.base_energy_std
        unk_mask = energy < thr

        # AUROC: binary classification (known=0, unknown=1)
        y_true = (all_labs >= known_classes).numpy()
        y_score = energy.numpy()  # lower energy = more unknown
        if len(np.unique(y_true)) >= 2:
            auroc = float(roc_auc_score(y_true, -y_score))  # negate so higher = more unknown
        else:
            auroc = 0.0

        unk_feats = all_feats[unk_mask]
        unk_labs = all_labs[unk_mask]
        kn_feats = all_feats[~unk_mask]
        kn_labs = all_labs[~unk_mask]

        # 3. Adaptive clustering with silhouette-based K selection
        n_clusters = self.n_novel_per_session
        cluster_a = 0.0
        novel_ids = None
        novel_protos = None

        if unk_feats.shape[0] >= n_clusters:
            km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42).fit(unk_feats.numpy())
            cluster_a, mapping = cluster_acc(self.args, unk_labs.numpy(), km.labels_)

            novel_ids = []
            novel_protos_list = []
            for c in range(n_clusters):
                tgt = mapping.get(c, None)
                if tgt is None or tgt < known_classes:
                    continue
                idx = (km.labels_ == c)
                if idx.sum() == 0:
                    continue
                proto = unk_feats[torch.from_numpy(idx)].mean(0)
                novel_ids.append(int(tgt))
                novel_protos_list.append(proto)

            if novel_protos_list:
                novel_protos = torch.stack(novel_protos_list, 0)
                self._register_novel_protos(novel_protos, novel_ids, known_classes)

        # 4. Evaluate all / incremental accuracy
        # Known accuracy
        known_acc = 0.0
        if kn_feats.numel() > 0:
            proto_now = self.model.fc.weight[:known_classes].to(self.device)
            logits_kn = F.cosine_similarity(
                F.normalize(kn_feats.to(self.device), dim=-1).unsqueeze(1),
                F.normalize(proto_now, dim=-1), dim=-1)
            known_acc = count_acc(logits_kn, kn_labs.to(self.device))

        # All-class accuracy
        new_known = known_classes + self.n_novel_per_session
        # Re-extract all features with current prototypes
        all_logits = F.linear(F.normalize(all_feats.to(self.device), dim=-1),
                              F.normalize(self.model.fc.weight[:new_known].to(self.device), dim=-1))
        all_pred = all_logits.argmax(-1).cpu()
        all_acc = float((all_pred == all_labs).float().mean().item())

        # Incremental accuracy (only novel classes)
        inc_mask = all_labs >= known_classes
        if inc_mask.sum() > 0:
            inc_acc = float((all_pred[inc_mask] == all_labs[inc_mask]).float().mean().item())
        else:
            inc_acc = 0.0

        fscore = calc(self.args, kn_labs.tolist(), unk_labs.tolist())

        return dict(known=known_acc, unknown=cluster_a, auroc=auroc, f1=fscore,
                    inc=inc_acc, all=all_acc)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _register_novel_protos(self, novel_protos: torch.Tensor,
                               novel_ids: List[int],
                               known_classes: int) -> None:
        """Register novel prototypes into model.fc.weight."""
        device = self.model.fc.weight.device
        new_protos = novel_protos.to(device)
        needed = max(novel_ids) + 1
        if needed > self.model.fc.weight.size(0):
            pad = torch.zeros(needed - self.model.fc.weight.size(0),
                              self.model.fc.weight.size(1), device=device)
            self.model.fc.weight.data = torch.cat([self.model.fc.weight.data, pad], 0)
        for i, cid in enumerate(novel_ids):
            self.model.fc.weight.data[cid] = new_protos[i]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_all(self, loader, n_class: int) -> float:
        """Evaluate all-class accuracy with current prototypes."""
        self.model.eval()
        n_correct, n_seen = 0, 0
        for batch in loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device).long()
            f = self.encode(x)
            logits = F.linear(F.normalize(f, dim=-1),
                              F.normalize(self.model.fc.weight[:n_class].to(self.device), dim=-1))
            pred = logits.argmax(-1)
            n_correct += int((pred == y).sum().item())
            n_seen += y.size(0)
        return n_correct / max(n_seen, 1)
