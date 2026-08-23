"""CEC — Continually Evolved Classifiers (Zhang et al., ICLR 2021).

Graph-based prototype evolution. We reuse a lightweight multi-head self
attention as the GAT-like evolver: prototypes interact through attention
to refresh each other at every new session, then cosine logits are used
for classification. The encoder is kept frozen — only prototypes evolve.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CILBase, cosine_logits, train_backbone_with_loss


class _ProtoEvolver(nn.Module):
    """Single-layer multi-head self-attention for prototype refresh."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, protos: torch.Tensor) -> torch.Tensor:
        # protos: [1, C, D]
        h, _ = self.attn(protos, protos, protos, need_weights=False)
        protos = self.norm1(protos + h)
        protos = self.norm2(protos + self.ff(protos))
        return protos


class CEC(CILBase):
    def __init__(self, model, args):
        super().__init__(model, args)
        dim = getattr(model, 'num_features', 512)
        self.evolver = _ProtoEvolver(dim, n_heads=getattr(args, 'cec_heads', 4))
        self.temperature = float(getattr(args, 'cec_temperature', 16.0))
        # start from the pre-trained base classifier weights
        self._protos = nn.Parameter(model.fc.weight.detach().clone(),
                                    requires_grad=False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        device = self._protos.device
        support_feats = support_feats.to(device)
        class_ids = list(class_ids)
        # mean feature = class prototype
        if support_feats.dim() == 3:  # [n_way, n_shot, D]
            new_protos = support_feats.mean(dim=1)
        else:
            assert support_feats.shape[0] == len(class_ids), \
                "support features must be aggregated to per-class prototypes"
            new_protos = support_feats
        cur = self._protos.data
        needed = max(class_ids) + 1
        if needed > cur.size(0):
            pad = torch.zeros(needed - cur.size(0), cur.size(1), device=device)
            cur = torch.cat([cur, pad], dim=0)
        for i, cid in enumerate(class_ids):
            cur[cid] = new_protos[i]
        # evolve prototypes via attention
        evolved = self.evolver(cur.unsqueeze(0)).squeeze(0)
        self._protos = nn.Parameter(evolved.detach(), requires_grad=False)

    # ------------------------------------------------------------------
    def classify(self, features: torch.Tensor, n_known: int) -> torch.Tensor:
        protos = self._protos[:n_known]
        return cosine_logits(features, protos, temperature=self.temperature)

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self._protos[:n_known].detach()

    # ------------------------------------------------------------------
    # Base training (standard cosine CE — CEC paper protocol)
    # ------------------------------------------------------------------
    def train_base(self, args, trainloader, log_path: Optional[str] = None) -> None:
        T = self.temperature

        def _loss(feats, w, y):
            logits = T * F.linear(F.normalize(feats, dim=-1),
                                  F.normalize(w, dim=-1))
            return F.cross_entropy(logits, y)

        train_backbone_with_loss(self.model, args, trainloader,
                                 loss_fn=_loss, tag='cec_base',
                                 log_path=log_path)
        self._protos = nn.Parameter(self.model.fc.weight.detach().clone(),
                                    requires_grad=False)
