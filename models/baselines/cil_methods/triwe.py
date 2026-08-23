"""Tri-WE — Tripartite Weight-Space Ensemble (Lee et al., CVPR 2025).

Core idea: whenever novel-class prototypes are added, the final classifier
weights are a convex combination of three snapshots:

    w_ensemble = alpha * w_base + beta * w_prev + gamma * w_current

- w_base   : original base classifier after stage-0 training (never changes)
- w_prev   : classifier weights *before* the current session update
- w_current: classifier weights *after* registering this session's novel prototypes

Normalized cosine logits are used for classification (no temperature).
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CILBase, cosine_logits, train_backbone_with_loss


class TriWE(CILBase):
    def __init__(self, model, args):
        super().__init__(model, args)
        dim = getattr(model, 'num_features', 512)
        # frozen snapshot of the *base* classifier weights (stage 0)
        self.register_buffer('_base_w', model.fc.weight.detach().clone())
        # running prototypes (will be updated at each session)
        self._protos = nn.Parameter(model.fc.weight.detach().clone(),
                                    requires_grad=False)
        # snapshot taken before the latest update
        self._prev_protos = None

        # interpolation coefficients (α + β + γ = 1)
        self.alpha = float(getattr(args, 'triwe_alpha', 0.20))
        self.beta  = float(getattr(args, 'triwe_beta', 0.30))
        self.gamma = float(getattr(args, 'triwe_gamma', 0.50))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        device = self._protos.device
        class_ids = list(class_ids)

        # 1. Save current prototypes as "previous" before updating
        self._prev_protos = self._protos.data.clone()

        # 2. Compute new prototypes from clustered features
        if support_feats.dim() == 3:  # [n_way, n_shot, D]
            new_protos = support_feats.mean(dim=1)
        else:
            new_protos = support_feats  # already per-class protos

        cur = self._protos.data
        needed = max(class_ids) + 1
        if needed > cur.size(0):
            pad = torch.zeros(needed - cur.size(0), cur.size(1), device=device)
            cur = torch.cat([cur, pad], dim=0)
        for i, cid in enumerate(class_ids):
            cur[cid] = new_protos[i].to(device)

        # 3. Tripartite weight-space ensemble
        #    base_w has shape [n_base, D]; it only covers base classes.
        #    For novel class rows we fall back to the current prototype.
        n_base = self._base_w.size(0)
        ensemble = torch.zeros_like(cur)
        ensemble[:n_base] = (self.alpha * self._base_w[:n_base] +
                             self.beta * self._prev_protos[:n_base] +
                             self.gamma * cur[:n_base])
        if cur.size(0) > n_base:
            ensemble[n_base:] = cur[n_base:]

        self._protos = nn.Parameter(ensemble.detach(), requires_grad=False)

    # ------------------------------------------------------------------
    def classify(self, features: torch.Tensor, n_known: int) -> torch.Tensor:
        protos = self._protos[:n_known]
        return cosine_logits(features, protos)  # temperature=1

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self._protos[:n_known].detach()

    # ------------------------------------------------------------------
    def train_base(self, args, trainloader, log_path: Optional[str] = None) -> None:
        """Standard cosine CE base training (identical to CEC)."""

        def _loss(feats, w, y):
            logits = F.linear(F.normalize(feats, dim=-1),
                              F.normalize(w, dim=-1))
            return F.cross_entropy(logits, y)

        train_backbone_with_loss(self.model, args, trainloader,
                                 loss_fn=_loss, tag='triwe_base',
                                 log_path=log_path)
        # Update base and running prototypes
        base_w = self.model.fc.weight.detach().clone()
        self._base_w = base_w.clone()
        self._protos = nn.Parameter(base_w.clone(), requires_grad=False)
