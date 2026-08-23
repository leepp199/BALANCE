"""Tri-WE: Tripartite Weight-Space Ensemble (CVPR 2025).

Reference: Lee et al., "Tripartite Weight-Space Ensemble for Few-Shot
Class-Incremental Learning", CVPR 2025.

Core algorithm:
1. Maintain three sets of classification head weights:
   - phi_0: base model head (from session 0, fixed)
   - phi_old: previous session's head
   - phi_all: current session's head

2. Tri-WE interpolation:
   For base classes: phi = alpha1*phi_0 + alpha2*phi_old + alpha3*phi_all
   For old inc classes: phi = alpha4*phi_old + alpha5*phi_all
   For new classes: phi = phi_all

3. Learnable alpha parameters via softmax normalization.

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


class TriWE(CILBase):
    def __init__(self, args):
        super().__init__(args)
        n_all = int(getattr(args, 'num_all', 100))
        dim = 512
        n_base = int(getattr(args, 'num_base', 80))

        # Own encoder — standard torchvision ResNet18
        self.model = AudioResNet(num_classes=n_all, pretrained=True,
                                 num_features=dim)

        self.register_buffer('_protos', torch.zeros(n_all, dim))
        if hasattr(self.model.fc, 'weight'):
            n_copy = min(self.model.fc.weight.size(0), n_all)
            self._protos[:n_copy] = self.model.fc.weight[:n_copy].detach()

        # Tri-WE: three sets of weights
        self.register_buffer('_phi_0', torch.zeros(n_all, dim))   # base, fixed
        self.register_buffer('_phi_old', torch.zeros(n_all, dim)) # previous
        self.register_buffer('_phi_all', torch.zeros(n_all, dim)) # current

        # Learnable interpolation coefficients (before softmax)
        self._alpha_base = nn.Parameter(torch.tensor([2.0, 1.0, 1.0]))
        self._alpha_old_inc = nn.Parameter(torch.tensor([2.0, 1.0]))

    # ==================================================================
    def train_base(self, args, trainloader,
                   log_path: Optional[str] = None) -> None:
        """Standard CE base training."""
        train_backbone_with_loss(
            self.model, args, trainloader,
            tag='triwe_base', log_path=log_path)
        n_base = int(getattr(args, 'num_base', 80))
        w = self.model.fc.weight[:n_base].detach()
        self._protos[:n_base] = w
        self._phi_0[:n_base] = w.clone()
        self._phi_old[:n_base] = w.clone()
        self._phi_all[:n_base] = w.clone()

    # ==================================================================
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        """Register novel prototypes via Tri-WE interpolation."""
        device = self._protos.device
        class_ids = list(class_ids)

        sf = support_feats.to(device)
        if sf.dim() == 3:
            new_protos = sf.mean(dim=1)
        else:
            new_protos = sf

        n_known_before = max(class_ids) + 1 - len(class_ids)

        # Update phi_all with new prototypes
        for i, cid in enumerate(class_ids):
            self._phi_all[cid] = new_protos[i]

        # Tri-WE interpolation
        alpha_base = F.softmax(self._alpha_base, dim=0)
        alpha_old_inc = F.softmax(self._alpha_old_inc, dim=0)

        for c in range(n_known_before + len(class_ids)):
            if c < int(getattr(self.args, 'num_base', 80)):
                # Base class: interpolate all three
                self._protos[c] = (
                    alpha_base[0] * self._phi_0[c] +
                    alpha_base[1] * self._phi_old[c] +
                    alpha_base[2] * self._phi_all[c])
            elif c < n_known_before:
                # Old incremental: interpolate phi_old + phi_all
                self._protos[c] = (
                    alpha_old_inc[0] * self._phi_old[c] +
                    alpha_old_inc[1] * self._phi_all[c])
            else:
                # New class: use phi_all directly
                self._protos[c] = self._phi_all[c]

        # Shift: old ← all for next session
        self._phi_old[:n_known_before + len(class_ids)] = \
            self._phi_all[:n_known_before + len(class_ids)].clone()

    # ==================================================================
    def classify(self, features: torch.Tensor,
                 n_known: int) -> torch.Tensor:
        return cosine_logits(features, self._protos[:n_known])

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self._protos[:n_known].detach()
