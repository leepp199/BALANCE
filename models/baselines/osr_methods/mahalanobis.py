"""Mahalanobis — Mahalanobis-distance open-set score (Lee et al., 2018).

Computes Mahalanobis distance from each sample to the nearest class centroid
using a shared covariance matrix estimated from features.

d_mahal(x, μ_c) = (x - μ_c)^T Σ^{-1} (x - μ_c)

Higher score (larger distance) → more likely unknown.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import OSRBase


class Mahalanobis(OSRBase):
    def __init__(self, args):
        super().__init__(args)
        self.reg = float(getattr(args, 'mahal_reg', 0.01))

    def score(self, features: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        # protos: [C_known, D] — class centroids in feature space
        # features: [N, D]
        f = F.normalize(features, dim=-1)
        p = F.normalize(protos, dim=-1)

        # Centered features (relative to mean of all features)
        feat_centered = f - f.mean(dim=0, keepdim=True)

        # Shared covariance with regularization
        cov = (feat_centered.t() @ feat_centered) / (feat_centered.size(0) - 1 + 1e-8)
        cov.diagonal().add_(self.reg)

        # Pseudo-inverse for numerical stability
        try:
            cov_inv = torch.linalg.pinv(cov)
        except Exception:
            cov_inv = torch.linalg.pinv(cov + self.reg * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype))

        # Mahalanobis distance to nearest class centroid
        diff = f.unsqueeze(1) - p.unsqueeze(0)  # [N, C, D]
        m_dist = (diff @ cov_inv) * diff  # [N, C, D]
        m_dist = m_dist.sum(dim=-1)  # [N, C]
        min_dist, _ = m_dist.min(dim=1)  # [N]

        # Higher distance → more unknown
        return min_dist
