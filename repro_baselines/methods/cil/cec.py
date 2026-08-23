"""CEC: Continually Evolved Classifiers (ICLR 2021).

Reference: Zhang et al., "Few-Shot Incremental Learning with Continually
Evolved Classifiers", ICLR 2021.

Core algorithm:
- A graph attention network evolves base class prototypes into new sessions.
- When novel class prototypes are added, the evolver refreshes ALL
  prototypes (base + novel) via multi-head self-attention.
- This mitigates catastrophic forgetting by letting novel prototypes
  attend to and update base prototypes.

Uses standard torchvision ResNet18 (ImageNet-pretrained), not MYNET.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from repro_baselines.models.base_encoder import AudioResNet
from ..base import CILBase, cosine_logits, train_backbone_with_loss


class _ProtoEvolver(nn.Module):
    """Single-layer multi-head self-attention for prototype evolution."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads,
                                          batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, protos: torch.Tensor) -> torch.Tensor:
        """protos: (K, D) → evolved: (K, D) with same shape."""
        x = protos.unsqueeze(0)  # (1, K, D)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out.squeeze(0) + protos)


class CEC(CILBase):
    def __init__(self, args):
        super().__init__(args)
        n_all = int(getattr(args, 'num_all', 100))
        dim = 512

        # Own encoder — standard torchvision ResNet18
        self.model = AudioResNet(num_classes=n_all, pretrained=True,
                                 num_features=dim)

        self.register_buffer('_protos', torch.zeros(n_all, dim))
        if hasattr(self.model.fc, 'weight'):
            n_copy = min(self.model.fc.weight.size(0), n_all)
            self._protos[:n_copy] = self.model.fc.weight[:n_copy].detach()

        # Graph attention evolver
        self.evolver = _ProtoEvolver(dim)

    # ==================================================================
    def train_base(self, args, trainloader,
                   log_path: Optional[str] = None) -> None:
        """Standard CE base training, then copy fc → prototypes."""
        T = float(getattr(getattr(args, 'network', args), 'temperature', 16.0))

        def _loss(feats, w, y):
            logits = T * F.linear(F.normalize(feats, dim=-1),
                                  F.normalize(w, dim=-1))
            return F.cross_entropy(logits, y)

        train_backbone_with_loss(
            self.model, args, trainloader, loss_fn=_loss,
            tag='cec_base', log_path=log_path)
        n_base = int(getattr(args, 'num_base', 80))
        self._protos[:n_base] = self.model.fc.weight[:n_base].detach()

    # ==================================================================
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        """Add novel prototypes, then evolve ALL prototypes via self-attn."""
        device = self._protos.device
        class_ids = list(class_ids)

        sf = support_feats.to(device)
        if sf.dim() == 3:
            new_protos = sf.mean(dim=1)
        else:
            new_protos = sf

        n_known = max(class_ids) + 1
        for i, cid in enumerate(class_ids):
            self._protos[cid] = new_protos[i]

        # Evolve all prototypes via graph attention
        active = self._protos[:n_known]
        evolved = self.evolver(active)
        self._protos[:n_known] = evolved

    # ==================================================================
    def classify(self, features: torch.Tensor,
                 n_known: int) -> torch.Tensor:
        return cosine_logits(features, self._protos[:n_known])

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self._protos[:n_known].detach()
