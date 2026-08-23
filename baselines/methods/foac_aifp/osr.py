"""FOAC-AIFP — Few-shot Open-set Audio Classification with
Attention Information-Fused Prototypes.

Simplified reimplementation. Uses attention-weighted prototype fusion:
  - PGFC (Prototype Generator for Few-shot Classes): attention over support samples
  - PGOC (Prototype Generator for Open-set Classes): open-set prototype from
    known-class prototypes with negative attention

Reference: IEEE TASLP 2026.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttentionFusion(nn.Module):
    """Attention-weighted prototype fusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim // 4)
        self.key = nn.Linear(dim, dim // 4)
        self.value = nn.Linear(dim, dim)

    def forward(self, support: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support: [K, D] support samples (few-shot).
        Returns:
            prototype: [D] attention-fused prototype.
        """
        K = support.size(0)
        q = self.query(support)    # [K, D/4]
        k = self.key(support)      # [K, D/4]
        v = self.value(support)    # [K, D]
        attn = torch.softmax((q @ k.T) / (q.size(-1) ** 0.5), dim=-1)  # [K, K]
        fused = (attn @ v).mean(dim=0)  # [D]
        return fused


class FOACAIFPOSR:
    """FOAC-AIFP OSR detector.

    Uses attention-fused prototypes for known/unknown discrimination.
    Score = -max_c attention_score(x, proto_c)
    Higher score → more likely unknown.
    """

    def __init__(self, temperature: float = 10.0, dim: int = 512):
        self.temperature = temperature
        self.fusion = _AttentionFusion(dim)
        self.refined_protos = None  # attention-refined prototypes
        self.threshold = None
        self.dim = dim

    def calibrate(self, features: torch.Tensor, prototypes: torch.Tensor,
                  labels: torch.Tensor = None, known_pct: float = 0.95) -> float:
        """Refine prototypes via attention fusion and calibrate threshold.

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

        # Refine each prototype using its class samples
        refined = []
        for c in range(C):
            if labels is not None:
                mask = labels == c
                if mask.sum() >= 2:
                    cls_feats = features[mask].to(device)
                    # Sample at most 50 for efficiency
                    if cls_feats.size(0) > 50:
                        idx = torch.randperm(cls_feats.size(0))[:50]
                        cls_feats = cls_feats[idx]
                    fused = self.fusion(cls_feats)
                    refined.append(fused)
                else:
                    refined.append(prototypes[c])
            else:
                refined.append(prototypes[c])
        self.refined_protos = torch.stack(refined, dim=0)

        scores = self.score(features, self.refined_protos)
        self.threshold = float(torch.quantile(scores, known_pct))
        return self.threshold

    def score(self, features: torch.Tensor,
              _prototypes: torch.Tensor = None) -> torch.Tensor:
        """Compute FOAC-AIFP scores.

        Uses attention between query features and refined prototypes.
        Higher score → more likely unknown.
        """
        if self.refined_protos is None:
            raise RuntimeError("Must call calibrate() before score()")
        device = features.device
        protos = self.refined_protos.to(device)

        # Multi-head style attention score
        feat_n = F.normalize(features, dim=1)
        proto_n = F.normalize(protos, dim=1)
        sim = self.temperature * (feat_n @ proto_n.T)  # [N, C]
        max_sim, _ = sim.max(dim=1)

        # Also compute attention-weighted score
        attn_weights = torch.softmax(sim / 2.0, dim=1)  # [N, C]
        attn_proto = attn_weights @ proto_n  # [N, D]
        attn_score = (feat_n * attn_proto).sum(dim=1) * self.temperature

        # Combined score: higher = more uncertain
        return -(max_sim + attn_score) / 2.0

    def is_known(self, features: torch.Tensor,
                 _prototypes: torch.Tensor = None) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before is_known()")
        return self.score(features) < self.threshold

    def get_scores_for_auroc(self, features: torch.Tensor,
                              _prototypes: torch.Tensor = None) -> torch.Tensor:
        return -self.score(features)
