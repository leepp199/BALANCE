"""Prototype-based OSR: max cosine similarity.

Simple baseline using maximum cosine similarity to known class prototypes.
Lower similarity = more likely unknown.

Reference: Standard prototype-based OSR baseline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class PrototypeOSR:
    """Prototype-based OSR using max cosine similarity.

    score = -max_c cosine_sim(x, proto_c)
    Higher score → more likely unknown.
    """

    def __init__(self, temperature: float = 10.0):
        self.temperature = temperature
        self.threshold = None

    def calibrate(self, features: torch.Tensor, prototypes: torch.Tensor,
                  known_pct: float = 0.95) -> float:
        """Calibrate threshold on known-class features.

        Args:
            features: [N, D] known-class features.
            prototypes: [C, D] classifier weight matrix.
            known_pct: percentile for threshold.

        Returns:
            threshold value.
        """
        scores = self.score(features, prototypes)
        # Higher score = more unknown, so threshold at (1-known_pct) quantile
        self.threshold = float(torch.quantile(scores, known_pct))
        return self.threshold

    def score(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute prototype scores.

        Higher score → more likely unknown.
        """
        feat_n = F.normalize(features, dim=1)
        proto_n = F.normalize(prototypes.to(features.device), dim=1)
        sim = self.temperature * (feat_n @ proto_n.T)  # [N, C]
        max_sim, _ = sim.max(dim=1)  # [N]
        return -max_sim  # negate so higher = more unknown

    def is_known(self, features: torch.Tensor,
                 prototypes: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True = known."""
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before is_known()")
        return self.score(features, prototypes) < self.threshold

    def get_scores_for_auroc(self, features: torch.Tensor,
                              prototypes: torch.Tensor) -> torch.Tensor:
        """Return scores where HIGHER = more likely KNOWN (for AUROC)."""
        return -self.score(features, prototypes)
