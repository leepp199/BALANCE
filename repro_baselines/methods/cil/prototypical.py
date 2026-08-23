"""ProtoNet: Prototypical Networks (NeurIPS 2017).

Reference: Snell et al., "Prototypical Networks for Few-Shot Learning",
NeurIPS 2017.

Uses standard torchvision ResNet18 (ImageNet-pretrained) as encoder,
not MYNET. Classification via cosine similarity to class prototypes.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from repro_baselines.models.base_encoder import AudioResNet
from ..base import CILBase, cosine_logits, train_backbone_with_loss


class ProtoNet(CILBase):
    """Prototypical Network baseline.

    Owns a standard AudioResNet encoder (no MYNET wrapper).
    """

    def __init__(self, args):
        super().__init__(args)
        n_all = int(getattr(args, 'num_all', 100))
        dim = 512

        # Own encoder — torchvision ResNet18, NOT MYNET
        self.model = AudioResNet(num_classes=n_all, pretrained=True,
                                 num_features=dim)
        self.register_buffer('_protos', torch.zeros(n_all, dim))

        # Copy initial fc weights as base prototypes
        if hasattr(self.model.fc, 'weight'):
            n_copy = min(self.model.fc.weight.size(0), n_all)
            self._protos.data[:n_copy] = self.model.fc.weight[:n_copy].detach()

    # ==================================================================
    def train_base(self, args, trainloader,
                   log_path: Optional[str] = None) -> None:
        """Standard CE base training via train_backbone_with_loss."""
        train_backbone_with_loss(
            self.model, args, trainloader,
            tag='prototypical_base', log_path=log_path)
        # Copy fc weights as prototypes
        n_base = int(getattr(args, 'num_base', 80))
        self._protos.data[:n_base] = self.model.fc.weight[:n_base].detach()

    # ==================================================================
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        """Register novel class prototypes."""
        device = self._protos.device
        class_ids = list(class_ids)
        if support_feats.dim() == 3:
            new_protos = support_feats.mean(dim=1)
        else:
            new_protos = support_feats
        for i, cid in enumerate(class_ids):
            self._protos.data[cid] = new_protos[i].to(device)

    # ==================================================================
    def classify(self, features: torch.Tensor,
                 n_known: int) -> torch.Tensor:
        return cosine_logits(features, self._protos[:n_known])

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self._protos[:n_known].detach()
