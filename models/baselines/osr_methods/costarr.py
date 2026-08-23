"""COSTARR — Consolidated Open Set Technique with Attenuation (ICCV 2025).

Reference: Rabinowitz et al., "COSTARR: Consolidated Open Set Technique with
Attenuation for Robust Recognition", ICCV 2025.

Core idea: during closed-set training, dimensions of the feature space that
help discriminate *known* from *unknown* tend to have their weights attenuated
because the cross-entropy loss only cares about separating known classes.
COSTARR recovers these attenuated dimensions to improve open-set detection.

Implementation (simplified for our frozen-encoder pipeline):
1. Compute the "attenuation profile" from the trained fc.weight matrix:
   dimensions with low inter-class variance are considered "attenuated".
2. At test time, amplify these attenuated dimensions by up-weighting them
   before computing the OSR score.
3. OSR score = max cosine similarity after attenuation-aware re-weighting.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import OSRBase


class COSTARR(OSRBase):
    """Open-set scoring via attenuation recovery.

    The detector is post-hoc (no extra training).  It works with any
    trained linear classifier.
    """

    def __init__(self, args):
        super().__init__(args)
        # amplification factor for attenuated dimensions
        self.amp_factor = float(getattr(args, 'costarr_amp', 2.0))
        # top-K% most attenuated dimensions to boost (default 20%)
        self.atten_ratio = float(getattr(args, 'costarr_ratio', 0.2))
        # computed after first call
        self._atten_mask = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_attenuation_mask(self, protos: torch.Tensor) -> torch.Tensor:
        """Identify the most 'attenuated' feature dimensions.

        Attenuation is measured by low inter-class variance of the
        classifier weights:
            atten[d] = var( fc.weight[:, d] )
        Dimensions with smallest variance are most attenuated.
        
        Returns a binary mask [D] marking dimensions to amplify.
        """
        w = F.normalize(protos, dim=-1)            # [n_class, D]
        var = w.var(dim=0, unbiased=False)          # [D]
        n_atten = max(1, int(var.numel() * self.atten_ratio))
        # Most attenuated = smallest variance
        _, idx = var.sort()
        mask = torch.zeros(var.numel(), dtype=torch.bool, device=var.device)
        mask[idx[:n_atten]] = True
        return mask

    # ------------------------------------------------------------------
    def score(self, features: torch.Tensor,
              protos: torch.Tensor) -> torch.Tensor:
        """OSR score: higher = more likely unknown.

        On the first call, computes the attenuation mask from ``protos``
        and caches it.
        """
        device = features.device
        protos = F.normalize(protos.to(device), dim=-1)

        if self._atten_mask is None or self._atten_mask.device != device:
            self._atten_mask = self._compute_attenuation_mask(protos)

        f = F.normalize(features, dim=-1)           # [B, D]
        # Amplify attenuated dimensions
        amp = torch.ones(f.size(-1), device=device)
        amp[self._atten_mask] = self.amp_factor
        f_amp = f * amp.unsqueeze(0)                # [B, D]
        f_amp = F.normalize(f_amp, dim=-1)

        # Score = - max similarity after attenuation recovery
        # (lower similarity = more unknown-like)
        sim = f_amp @ protos.t()                    # [B, n_class]
        max_sim, _ = sim.max(dim=-1)
        # Invert so higher = more unknown
        return -max_sim
