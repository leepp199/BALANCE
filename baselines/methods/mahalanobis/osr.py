"""Mahalanobis distance-based OSR detector.

Computes per-class means and shared covariance from base training data,
then uses minimum Mahalanobis distance to any known class as the
familiarity score. Larger distance = more likely unknown.

Reference: Lee et al., "A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018.
"""
from __future__ import annotations

import torch


class MahalanobisOSR:
    """Mahalanobis distance-based open-set recognition scorer.

    score = min_c (x - mu_c)^T Sigma^{-1} (x - mu_c)
    Higher score → more likely unknown.
    """

    def __init__(self, eps: float = 0.001):
        self.eps = eps
        self.means = None       # [C, D]
        self.inv_cov = None     # [D, D]
        self.threshold = None

    def calibrate(self, features: torch.Tensor, labels: torch.Tensor,
                  known_pct: float = 0.95) -> float:
        """Fit per-class means and shared precision matrix from known data.

        Args:
            features: [N, D] known-class features.
            labels: [N] class indices.
            known_pct: percentile for threshold.

        Returns:
            threshold value.
        """
        device = features.device
        num_classes = int(labels.max().item()) + 1
        D = features.size(1)

        means_list = []
        all_centered = []
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                means_list.append(torch.zeros(D, device=device))
                continue
            cls_feats = features[mask]
            mean = cls_feats.mean(dim=0)
            means_list.append(mean)
            all_centered.append(cls_feats - mean)

        self.means = torch.stack(means_list, dim=0)
        centered = torch.cat(all_centered, dim=0)
        n = centered.size(0)
        cov = (centered.T @ centered) / max(n - 1, 1)
        cov += self.eps * torch.eye(D, device=device)
        self.inv_cov = torch.linalg.pinv(cov)

        scores = self.score(features, None)
        self.threshold = float(torch.quantile(scores, known_pct))
        return self.threshold

    def score(self, features: torch.Tensor,
              _prototypes: torch.Tensor = None) -> torch.Tensor:
        """Compute Mahalanobis distances. Higher = more unknown."""
        if self.means is None or self.inv_cov is None:
            raise RuntimeError("Must call calibrate() before score()")
        device = features.device
        means = self.means.to(device)
        inv_cov = self.inv_cov.to(device)
        diff = features.unsqueeze(1) - means.unsqueeze(0)  # [N, C, D]
        dists = (diff @ inv_cov * diff).sum(dim=2)          # [N, C]
        return dists.min(dim=1)[0]                           # [N]

    def is_known(self, features: torch.Tensor,
                 _prototypes: torch.Tensor = None) -> torch.Tensor:
        """Boolean mask: True = known."""
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before is_known()")
        return self.score(features) < self.threshold

    def get_scores_for_auroc(self, features: torch.Tensor,
                              _prototypes: torch.Tensor = None) -> torch.Tensor:
        """Return scores where HIGHER = more likely KNOWN (for AUROC)."""
        return -self.score(features)
