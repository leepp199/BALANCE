"""NCI — Nearest-Class Inconsistency for open-set detection.

For each sample, collect the top-k nearest class prototypes; inconsistency
is defined as the ratio  dist(top1) / mean_dist(top2..k).  Ratios close
to 1 mean the sample is roughly equidistant to several classes → unknown.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import OSRBase


class NCI(OSRBase):
    def __init__(self, args):
        super().__init__(args)
        self.k = int(getattr(args, 'nci_topk', 5))

    def score(self, features: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        f = F.normalize(features, dim=-1)
        p = F.normalize(protos, dim=-1)
        # distance = 1 - cosine
        dist = 1.0 - f @ p.t()
        k = min(self.k, dist.size(1))
        vals, _ = torch.topk(dist, k=k, dim=1, largest=False)  # k nearest
        top1 = vals[:, 0].clamp(min=1e-8)
        rest = vals[:, 1:].mean(dim=1).clamp(min=1e-8)
        # ratio → 1 means ambiguous (unknown); ratio << 1 means confident
        return top1 / rest
