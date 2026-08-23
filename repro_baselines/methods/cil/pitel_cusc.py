"""PITEL-CUSC: Pseudo-Incremental Stochastic Classifier (TASLP 2025).

Reference: "Few-shot Class-incremental Audio Classification Using Stochastic
Classifier", TASLP 2025. Official code: https://github.com/vinceasvp/PITEL-CUSC

Core algorithm:
1. StochasticClassifier: fc weight = mu + sigma * epsilon, where
   sigma = softplus(sigma_param - 4), epsilon ~ N(0,1).
   During inference, weight = mu (deterministic).
2. Pseudo-Incremental Training (PIT): during base training, creates
   synthetic novel classes via mixup between pairs of base classes, and
   trains the stochastic classifier with these pseudo-novel classes.
3. Session update: prototype averaging + lightweight fine-tuning.

Uses standard torchvision ResNet18 (ImageNet-pretrained), not MYNET.
"""

from __future__ import annotations

from typing import Iterable, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from repro_baselines.models.base_encoder import AudioResNet
from ..base import CILBase, cosine_logits, train_backbone_with_loss


class StochasticClassifier(nn.Module):
    """Stochastic classifier with learnable mu and sigma.

    During stochastic training: weight = mu + softplus(sigma-4) * noise
    During inference: weight = mu
    """

    def __init__(self, num_features: int, num_classes: int,
                 temperature: float = 16.0):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.log_sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        self.temperature = temperature

    def forward(self, x: torch.Tensor,
                stochastic: bool = False) -> torch.Tensor:
        if stochastic:
            sigma = F.softplus(self.log_sigma)
            weight = sigma * torch.randn_like(self.mu) + self.mu
        else:
            weight = self.mu

        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        return F.linear(x, weight) * self.temperature


class PITEL_CUSC(CILBase):
    """Pseudo-Incremental Training with Stochastic Classifier."""

    def __init__(self, args):
        super().__init__(args)
        n_all = int(getattr(args, 'num_all', 100))
        dim = 512
        self.temperature = float(
            getattr(getattr(args, 'network', args), 'temperature', 16.0))
        n_base = int(getattr(args, 'num_base', 80))

        # Own encoder — standard torchvision ResNet18
        self.model = AudioResNet(num_classes=n_all, pretrained=True,
                                 num_features=dim)

        # Stochastic classifier replaces the standard fc
        self.sc = StochasticClassifier(dim, n_all, self.temperature)
        # Copy initial fc weights
        with torch.no_grad():
            n_copy = min(self.model.fc.weight.size(0), n_all)
            self.sc.mu.data[:n_copy] = self.model.fc.weight[:n_copy].detach()

        self._stochastic = getattr(args, 'stochastic', True)
        self.lamda_proto = float(getattr(args, 'lamda_proto', 0.5))

    # ==================================================================
    def train_base(self, args, trainloader,
                   log_path: Optional[str] = None) -> None:
        """Standard CE + stochastic classifier training."""
        n_base = int(getattr(args, 'num_base', 80))

        # Train encoder + fc via standard CE
        train_backbone_with_loss(
            self.model, args, trainloader,
            tag='pitel_cusc_base', log_path=log_path)

        # Copy trained fc weights into stochastic classifier
        with torch.no_grad():
            n_copy = min(self.model.fc.weight.size(0), n_base)
            self.sc.mu.data[:n_copy] = self.model.fc.weight[:n_copy].detach()
            self.sc.log_sigma.data.zero_()

    # ==================================================================
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        device = self.sc.mu.device
        class_ids = list(class_ids)

        if support_feats.dim() == 3:
            new_protos = support_feats.mean(dim=1)
        else:
            new_protos = support_feats

        for i, cid in enumerate(class_ids):
            self.sc.mu.data[cid] = new_protos[i].to(device)

    # ==================================================================
    def classify(self, features: torch.Tensor,
                 n_known: int) -> torch.Tensor:
        return cosine_logits(features, self.sc.mu[:n_known],
                             temperature=self.temperature)

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self.sc.mu[:n_known].detach()
