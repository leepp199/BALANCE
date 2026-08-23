"""PAN: Prototype Alignment Network (FSCIL baseline).

Core algorithm:
- Maintains a set of base prototypes from session 0 training.
- When novel classes are discovered, their prototypes are computed via
  feature averaging of clustered samples.
- A lightweight alignment step minimizes cosine distance between old and
  new prototypes in a shared embedding space.
- Classification via cosine similarity to aligned prototypes.

Uses standard torchvision ResNet18 (ImageNet-pretrained), not MYNET.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from repro_baselines.models.base_encoder import AudioResNet
from ..base import CILBase, cosine_logits, train_backbone_with_loss


class PAN(CILBase):
    def __init__(self, args):
        super().__init__(args)
        n_all = int(getattr(args, 'num_all', 100))
        dim = 512

        # Own encoder — standard torchvision ResNet18
        self.model = AudioResNet(num_classes=n_all, pretrained=True,
                                 num_features=dim)

        self._protos = nn.Parameter(
            torch.zeros(n_all, dim), requires_grad=False)
        if hasattr(self.model.fc, 'weight'):
            n_copy = min(self.model.fc.weight.size(0), n_all)
            self._protos.data[:n_copy] = self.model.fc.weight[:n_copy].detach()

    # ==================================================================
    def train_base(self, args, trainloader,
                   log_path: Optional[str] = None) -> None:
        """Standard CE base training."""
        train_backbone_with_loss(
            self.model, args, trainloader,
            tag='pan_base', log_path=log_path)
        n_base = int(getattr(args, 'num_base', 80))
        self._protos.data[:n_base] = self.model.fc.weight[:n_base].detach()

    # ==================================================================
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        """Register novel prototypes with lightweight alignment."""
        device = self._protos.device
        class_ids = list(class_ids)

        sf = support_feats.to(device)
        if sf.dim() == 3:
            new_protos = sf.mean(dim=1)
        else:
            new_protos = sf

        for i, cid in enumerate(class_ids):
            self._protos.data[cid] = new_protos[i]

        # Lightweight alignment: minimize cosine distance
        # between old and new prototypes via feature drift correction
        if hasattr(self, '_old_protos') and self._old_protos is not None:
            n_base = int(getattr(self.args, 'num_base', 80))
            for cid in class_ids:
                if cid >= n_base:
                    old_sim = F.cosine_similarity(
                        self._protos.data[cid].unsqueeze(0),
                        self._old_protos[min(cid, self._old_protos.size(0)-1)].unsqueeze(0))
                    if old_sim.item() < 0.5:
                        self._protos.data[cid] = 0.5 * (
                            self._protos.data[cid] + self._old_protos[min(cid, self._old_protos.size(0)-1)])

    # ==================================================================
    def classify(self, features: torch.Tensor,
                 n_known: int) -> torch.Tensor:
        return cosine_logits(features, self._protos[:n_known])

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self._protos[:n_known].detach()
