"""DNPG — Diversified Negative Prototypes Generator for OSR.

Generates diversified negative prototypes from known-class features.
For each known class, creates K negative prototypes by perturbing features
in directions away from all known classes, then uses the minimum distance
to any negative prototype as the unknown score.

Reference: DNPG, ACM MM 2024.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class DNPGOSR:
    """Diversified Negative Prototypes Generator.

    Generates negative prototypes that are pushed away from all known classes.
    Score = -min_c cosine_sim(x, neg_proto_c).
    Higher score → more likely unknown.
    """

    def __init__(self, n_neg_per_class: int = 3, noise_scale: float = 0.3,
                 temperature: float = 10.0):
        self.n_neg = n_neg_per_class
        self.noise_scale = noise_scale
        self.temperature = temperature
        self.neg_protos = None  # [n_neg * C, D]
        self.threshold = None

    def _generate_negative_prototypes(self, prototypes: torch.Tensor,
                                       features: torch.Tensor,
                                       labels: torch.Tensor) -> torch.Tensor:
        """Generate diversified negative prototypes.

        For each known class c:
          1. Take the mean feature mu_c
          2. Generate K perturbations in orthogonal directions
          3. Push them away from all mu_c via gradient-free optimization
        """
        device = prototypes.device
        C, D = prototypes.shape

        # Normalize prototypes
        proto_n = F.normalize(prototypes, dim=1)

        neg_list = []
        for c in range(C):
            # Base direction: the class prototype itself
            base = proto_n[c].clone()

            # Generate random orthogonal directions
            for k in range(self.n_neg):
                # Random direction
                rand_dir = torch.randn(D, device=device)
                # Make orthogonal to base
                rand_dir = rand_dir - (rand_dir @ base) * base / (base @ base + 1e-8)
                rand_dir = F.normalize(rand_dir, dim=0)

                # Create negative prototype: move away from this class
                # and also away from all other classes
                neg = base + self.noise_scale * rand_dir

                # Push away from ALL classes simultaneously
                cos_all = neg @ proto_n.T  # [C]
                # Gradient to reduce max cosine
                push_strength = torch.sigmoid(cos_all * 5.0).mean()
                neg = neg - push_strength * 0.1 * (neg @ proto_n.T @ proto_n)
                neg = F.normalize(neg, dim=0)

                neg_list.append(neg)

        return torch.stack(neg_list, dim=0)

    def calibrate(self, features: torch.Tensor, prototypes: torch.Tensor,
                  labels: torch.Tensor = None, known_pct: float = 0.95) -> float:
        """Generate negative prototypes and calibrate threshold.

        Args:
            features: [N, D] known-class features.
            prototypes: [C, D] known class prototypes.
            labels: [N] class indices (optional, for generation).
            known_pct: percentile for threshold.

        Returns:
            threshold value.
        """
        device = prototypes.device
        # Generate negative prototypes if labels provided
        if labels is not None:
            self.neg_protos = self._generate_negative_prototypes(
                prototypes, features.to(device), labels.to(device))
        else:
            # Simple: use prototypes perturbed in random directions
            C, D = prototypes.shape
            proto_n = F.normalize(prototypes, dim=1)
            negs = []
            for c in range(C):
                for k in range(self.n_neg):
                    noise = F.normalize(torch.randn(D, device=device), dim=0)
                    neg = proto_n[c] + self.noise_scale * noise
                    negs.append(F.normalize(neg, dim=0))
            self.neg_protos = torch.stack(negs, dim=0)

        scores = self.score(features, prototypes)
        self.threshold = float(torch.quantile(scores, known_pct))
        return self.threshold

    def score(self, features: torch.Tensor,
              _prototypes: torch.Tensor = None) -> torch.Tensor:
        """Compute DNPG scores.

        Score = -max cosine similarity to negative prototypes.
        Higher score → more likely unknown.
        """
        if self.neg_protos is None:
            raise RuntimeError("Must call calibrate() before score()")
        device = features.device
        neg = F.normalize(self.neg_protos.to(device), dim=1)
        feat_n = F.normalize(features, dim=1)
        # Cosine similarity to negative prototypes
        sim = self.temperature * (feat_n @ neg.T)  # [N, K*C]
        max_sim, _ = sim.max(dim=1)  # [N]
        # High sim to neg = likely unknown
        return max_sim

    def is_known(self, features: torch.Tensor,
                 _prototypes: torch.Tensor = None) -> torch.Tensor:
        """Boolean mask: True = known."""
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before is_known()")
        return self.score(features) < self.threshold

    def get_scores_for_auroc(self, features: torch.Tensor,
                              prototypes: torch.Tensor = None) -> torch.Tensor:
        """Return scores where HIGHER = more likely KNOWN (for AUROC)."""
        return -self.score(features)
