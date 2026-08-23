"""Energy-based OSR detector.

E(x) = -T * logsumexp(f(x)_i / T)
Lower energy = more likely in-distribution (known).
Higher energy = more likely OOD (unknown).

Reference: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class EnergyOSR:
    """Energy-based open-set recognition scorer.

    score = -T * logsumexp(logits / T)
    Higher score → more likely unknown.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.threshold = None

    def calibrate(self, features: torch.Tensor, prototypes: torch.Tensor,
                  known_pct: float = 0.95) -> float:
        """Calibrate threshold on known-class features.

        Args:
            features: [N, D] known-class features.
            prototypes: [C, D] classifier weight matrix.
            known_pct: percentile for threshold (fraction of known samples accepted).

        Returns:
            threshold value.
        """
        scores = self.score(features, prototypes)
        self.threshold = float(torch.quantile(scores, known_pct))
        return self.threshold

    def score(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute energy scores. Higher = more unknown."""
        proto = prototypes.to(features.device)
        logits = F.linear(features, proto) / self.temperature
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return energy

    def is_known(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True = known."""
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before is_known()")
        return self.score(features, prototypes) < self.threshold

    def get_scores_for_auroc(self, features: torch.Tensor,
                              prototypes: torch.Tensor) -> torch.Tensor:
        """Return scores where HIGHER = more likely KNOWN (for AUROC)."""
        return -self.score(features, prototypes)
