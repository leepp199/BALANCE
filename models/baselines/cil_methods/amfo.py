"""AMFO — Angular Margin Few-shot Optimization (audio variant).

Uses an additive angular margin on cosine logits and *freezes* the base
prototypes once base training is done; only novel prototypes are added
at each incremental session.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CILBase, train_backbone_with_loss


class AMFO(CILBase):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.margin = float(getattr(args, 'amfo_margin', 0.2))
        self.scale = float(getattr(args, 'amfo_scale', 16.0))
        # frozen base prototypes + appended novel prototypes
        self._protos = nn.Parameter(model.fc.weight.detach().clone(),
                                    requires_grad=False)
        self._base_size = int(getattr(args, 'base_class', 80))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        device = self._protos.device
        support_feats = support_feats.to(device)
        class_ids = list(class_ids)
        if support_feats.dim() == 3:
            new_protos = support_feats.mean(dim=1)
        else:
            new_protos = support_feats
        cur = self._protos.data.clone()
        needed = max(class_ids) + 1
        if needed > cur.size(0):
            pad = torch.zeros(needed - cur.size(0), cur.size(1), device=device)
            cur = torch.cat([cur, pad], dim=0)
        for i, cid in enumerate(class_ids):
            if cid < self._base_size:
                continue  # base prototypes are frozen
            cur[cid] = F.normalize(new_protos[i], dim=0) * new_protos[i].norm()
        self._protos = nn.Parameter(cur, requires_grad=False)

    # ------------------------------------------------------------------
    def classify(self, features: torch.Tensor, n_known: int,
                 labels: torch.Tensor = None) -> torch.Tensor:
        protos = self._protos[:n_known]
        f = F.normalize(features, dim=-1)
        p = F.normalize(protos, dim=-1)
        cos = f @ p.t()  # [B, C]
        if labels is None:  # inference: plain scaled cosine
            return self.scale * cos
        # training: additive angular margin on the ground-truth column
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        m_hot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), self.margin)
        logits = torch.cos(theta + m_hot)
        return self.scale * logits

    # ------------------------------------------------------------------
    # Base training (ArcFace-style additive margin)
    # ------------------------------------------------------------------
    def train_base(self, args, trainloader, log_path: Optional[str] = None) -> None:
        margin = self.margin
        scale = self.scale

        def _loss(feats, w, y):
            f = F.normalize(feats, dim=-1)
            p = F.normalize(w, dim=-1)
            cos = f @ p.t()
            theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
            m_hot = torch.zeros_like(cos).scatter_(1, y.view(-1, 1), margin)
            logits = scale * torch.cos(theta + m_hot)
            return F.cross_entropy(logits, y)

        train_backbone_with_loss(self.model, args, trainloader,
                                 loss_fn=_loss, tag='amfo_base',
                                 log_path=log_path)
        self._protos = nn.Parameter(self.model.fc.weight.detach().clone(),
                                    requires_grad=False)
