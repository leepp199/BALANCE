"""MLS — Maximum Logit Score (Vaze et al., ICLR 2022).

unknown_score = -max_c ( scaled_cosine(feat, proto_c) )

Higher score → more likely unknown.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import OSRBase


class MLS(OSRBase):
    def __init__(self, args):
        super().__init__(args)
        self.scale = float(getattr(args, 'mls_scale', 16.0))

    def score(self, features: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        f = F.normalize(features, dim=-1)
        p = F.normalize(protos, dim=-1)
        logits = self.scale * (f @ p.t())
        max_logit, _ = logits.max(dim=1)
        return -max_logit
