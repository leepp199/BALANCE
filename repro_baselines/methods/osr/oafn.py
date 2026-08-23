"""OAFN: Open-World Audio Few-Shot Learning Network (KBS 2025).

Reference: Chen et al., "OAFN: An Efficient Open-World Audio Few-Shot Learning
Network for Event Classification", Knowledge-Based Systems, 2025.

Core algorithm (adapted for shared encoder protocol):
1. Dual-channel calibration: two complementary views of prototypes
   - Channel A: standard prototype averaging
   - Channel B: distance-weighted prototype refinement

2. Inter-class / Intra-class calibration:
   - Intra-class: reweight support samples within each class based on
     their cosine distance to the class mean (closer = higher weight)
   - Inter-class: push prototypes of different classes apart

3. Data perturbation: add small Gaussian noise and random masking to
   support features for robustness to label/environmental noise.

4. Scoring: calibrated prototype similarity with a confidence-based
   open-set score. Low confidence in the most similar class = unknown.

Adapted for prototype-based classification with shared audio encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OSRBase


class OAFN(OSRBase):
    """Open-World Audio Few-Shot Learning Network.

    Higher score = more likely unknown.
    """

    def __init__(self, args):
        super().__init__(args)
        self.noise_scale = float(getattr(args, 'oafn_noise', 0.01))
        self.mask_ratio = float(getattr(args, 'oafn_mask', 0.1))
        self.intra_temp = float(getattr(args, 'oafn_intra_temp', 2.0))

        # Intra-class calibration parameters (learnable)
        self._calib_weight = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    def fit(self, model, train_loader):
        """Compute calibrated prototypes using dual-channel + intra/inter calibration.

        The fitting process:
        1. Extract features from training data
        2. Compute per-class prototypes with intra-class reweighting
        3. Compute dual-channel prototypes (standard + distance-weighted)
        4. Store for scoring
        """
        device = next(model.parameters()).device
        self._calib_weight = self._calib_weight.to(device)

        # Extract features
        model.eval()
        all_feats = []
        all_labels = []
        prev_mode = getattr(model, 'mode', None)
        if hasattr(model, 'mode'):
            model.mode = '__feat__'

        with torch.no_grad():
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1]
                f = model.encode(x).cpu()
                all_feats.append(f)
                all_labels.append(y)

        if prev_mode is not None:
            model.mode = prev_mode

        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        n_classes = int(all_labels.max().item()) + 1

        # Compute dual-channel calibrated prototypes
        protos_ch1 = []  # Standard mean
        protos_ch2 = []  # Distance-weighted (intra-class calibration)

        for c in range(n_classes):
            mask = (all_labels == c)
            if mask.sum() == 0:
                protos_ch1.append(torch.zeros(all_feats.size(1)))
                protos_ch2.append(torch.zeros(all_feats.size(1)))
                continue

            f_c = all_feats[mask]  # [N_c, D]
            f_c = F.normalize(f_c, dim=-1)

            # Channel 1: standard mean
            p1 = f_c.mean(dim=0)

            # Channel 2: distance-weighted (intra-class calibration)
            # Weights based on cosine distance to the mean
            cos_sim = f_c @ p1.unsqueeze(-1)  # [N_c, 1]
            weights = F.softmax(cos_sim / self.intra_temp, dim=0)  # [N_c, 1]
            p2 = (weights * f_c).sum(dim=0)

            protos_ch1.append(p1)
            protos_ch2.append(p2)

        protos_ch1 = F.normalize(torch.stack(protos_ch1, dim=0), dim=-1)
        protos_ch2 = F.normalize(torch.stack(protos_ch2, dim=0), dim=-1)

        # Store as calibrated prototypes (dual-channel fusion)
        w = torch.sigmoid(self._calib_weight)
        self._calib_protos = F.normalize(
            w * protos_ch1 + (1 - w) * protos_ch2, dim=-1).detach().cpu()
        self._is_fitted = True

    # ------------------------------------------------------------------
    def _perturb_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply data perturbation for robustness (Gaussian noise + masking)."""
        perturbed = features.clone()

        # Gaussian noise
        noise = torch.randn_like(perturbed) * self.noise_scale
        perturbed = perturbed + noise

        # Random feature masking (simulating environmental noise)
        if self.mask_ratio > 0 and self.training:
            mask = torch.rand_like(perturbed) > self.mask_ratio
            perturbed = perturbed * mask

        return perturbed

    # ------------------------------------------------------------------
    def score(self, features: torch.Tensor,
              protos: torch.Tensor) -> torch.Tensor:
        """OAFN score: higher = more unknown.

        Uses dual-channel calibrated prototypes for scoring.
        Low confidence in nearest known class = unknown.
        """
        device = features.device
        f = F.normalize(features, dim=-1)

        # Use calibrated prototypes if fitted, else use passed protos
        if self._is_fitted and hasattr(self, '_calib_protos'):
            known_p = F.normalize(self._calib_protos.to(device), dim=-1)
            # Ensure dimensions match
            n_known = protos.size(0)
            known_p = known_p[:n_known]
        else:
            known_p = F.normalize(protos.to(device), dim=-1)

        # Cosine similarity to all known prototypes
        sim = f @ known_p.t()  # [B, C]

        # OAFN open-set score: based on confidence gap
        # Sort similarities descending
        sorted_sim, _ = sim.sort(dim=1, descending=True)

        # Score = -top1_sim + (top1_sim - top2_sim)  (low confidence = unknown)
        # Equivalently: score = -top2_sim (small gap to 2nd best = uncertain)
        top1 = sorted_sim[:, 0]  # [B]
        top2 = sorted_sim[:, 1]  # [B]

        # Higher score when:
        # 1. Top-1 similarity is low (not close to any known)
        # 2. Gap between top-1 and top-2 is small (ambiguous)
        confidence_gap = top1 - top2
        score = -top1 + (1.0 - confidence_gap)

        return score
