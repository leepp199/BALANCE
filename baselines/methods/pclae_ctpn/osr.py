"""PCLAE-CTPN — Pre-trained Contrastive Language-Audio Encoder +
Collaborative Prototype Network.

Simplified reimplementation. Uses:
  - Prototype compactness loss (pull same-class together)
  - Open-space loss (push different classes apart)
  - Per-class radius for known/unknown decision

The OSR score is based on the margin between nearest-class distance
and the class-specific radius.

Reference: IEEE TSP 2025.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class PCLAECTPNOSR:
    """PCLAE-CTPN OSR detector.

    Uses class-specific radius for open-set detection.
    Score = distance_to_nearest_class - radius_of_nearest_class.
    Positive score → outside the known-class boundary → unknown.
    """

    def __init__(self, radius_percentile: float = 0.90,
                 temperature: float = 10.0):
        self.radius_pct = radius_percentile
        self.temperature = temperature
        self.class_radii = None  # [C]
        self.prototypes = None   # [C, D]
        self.threshold = None

    def calibrate(self, features: torch.Tensor, prototypes: torch.Tensor,
                  labels: torch.Tensor = None, known_pct: float = 0.95) -> float:
        """Compute per-class radii and calibrate threshold.

        Args:
            features: [N, D] known-class features.
            prototypes: [C, D] known class prototypes.
            labels: [N] class indices.
            known_pct: percentile for threshold.

        Returns:
            threshold value.
        """
        device = prototypes.device
        C = prototypes.size(0)
        self.prototypes = F.normalize(prototypes, dim=1)

        radii = []
        for c in range(C):
            if labels is not None:
                mask = labels == c
                if mask.sum() > 0:
                    cls_feats = F.normalize(features[mask].to(device), dim=1)
                    proto = self.prototypes[c].unsqueeze(0)  # [1, D]
                    # Cosine distance to class prototype
                    cos_sim = (cls_feats @ proto.T).squeeze(1)  # [N_c]
                    dists = 1.0 - cos_sim  # cosine distance
                    # Radius = percentile of distances
                    r = torch.quantile(dists, self.radius_pct).item()
                    radii.append(r)
                else:
                    radii.append(0.5)
            else:
                # Default: radius based on prototype separation
                other = torch.cat([self.prototypes[:c], self.prototypes[c + 1:]], dim=0)
                min_dist = 1.0 - (self.prototypes[c] @ other.T).max()
                radii.append(float(min_dist.item()) * 0.5)

        self.class_radii = torch.tensor(radii, device=device)

        scores = self.score(features, self.prototypes)
        self.threshold = float(torch.quantile(scores, known_pct))
        return self.threshold

    def score(self, features: torch.Tensor,
              _prototypes: torch.Tensor = None) -> torch.Tensor:
        """Compute PCLAE-CTPN scores.

        Score = min_c (dist(x, proto_c) - radius_c)
        Higher score → more likely unknown.
        """
        if self.prototypes is None or self.class_radii is None:
            raise RuntimeError("Must call calibrate() before score()")
        device = features.device
        protos = self.prototypes.to(device)
        radii = self.class_radii.to(device)

        feat_n = F.normalize(features, dim=1)
        cos_sim = feat_n @ protos.T  # [N, C]
        dists = 1.0 - cos_sim  # cosine distance [N, C]

        # Nearest class
        min_dists, nearest = dists.min(dim=1)  # [N]
        nearest_radii = radii[nearest]  # [N]

        # Score = distance - radius of nearest class
        return min_dists - nearest_radii

    def is_known(self, features: torch.Tensor,
                 _prototypes: torch.Tensor = None) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before is_known()")
        return self.score(features) < self.threshold

    def get_scores_for_auroc(self, features: torch.Tensor,
                              _prototypes: torch.Tensor = None) -> torch.Tensor:
        return -self.score(features)
