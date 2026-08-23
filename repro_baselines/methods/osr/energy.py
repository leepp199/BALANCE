"""Energy-based open-set score (Liu et al., ICLR 2020).

energy = -T * logsumexp(logits / T)
Higher energy = more unknown.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import OSRBase


class Energy(OSRBase):
    def __init__(self, args):
        super().__init__(args)
        self.scale = float(getattr(args, 'energy_scale', 16.0))
        self.temperature = float(getattr(args, 'energy_temperature', 1.0))

    def score(self, features: torch.Tensor,
              protos: torch.Tensor) -> torch.Tensor:
        f = F.normalize(features, dim=-1)
        p = F.normalize(protos, dim=-1)
        logits = self.scale * (f @ p.t())
        energy = -self.temperature * torch.logsumexp(
            logits / self.temperature, dim=1)
        return energy
