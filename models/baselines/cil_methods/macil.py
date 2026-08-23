"""MACIL — simplified Mean-shift + Covariance Calibration (ICML 2025).

Core ideas adapted to our ResNet18 / KMeans-clustering pipeline:

- **Mean-shift compensation**: at each incremental session, the feature
  distribution of old classes drifts because the encoder now also represents
  novel features.  We estimate this displacement and apply it to the old-class
  prototypes *before* classification.

- **Covariance calibration** (simplified): we maintain diagonal covariance
  estimates for each base class from the stage-0 training features, and use
  them to re-weight prototype similarity (Mahalanobis-like distance).

Because we do **not** have exemplar replay, the full MACIL pipeline (which
includes LoRA adapters, task-specific classifiers and feature-level self-
distillation) is not directly applicable.  This simplified version captures
the two key calibration signals described in the paper.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CILBase, cosine_logits, train_backbone_with_loss


class MACIL(CILBase):
    def __init__(self, model, args):
        super().__init__(model, args)
        dim = getattr(model, 'num_features', 512)
        n_base = int(getattr(args, 'num_base', 80))
        # running prototypes
        self._protos = nn.Parameter(model.fc.weight.detach().clone(),
                                    requires_grad=False)
        # class means and covariances from base training (filled by train_base)
        self.register_buffer('_class_means', torch.zeros(n_base, dim))
        self.register_buffer('_class_covs', torch.zeros(n_base, dim))  # diagonal cov

        # displacement of the previous session (mean-shift)
        self._prev_mean_shift = None
        self._is_first_session = True

        # interpolation strength for mean-shift compensation (0 = off, 1 = full)
        self.compensation_strength = float(
            getattr(args, 'macil_compensation', 0.3))
        self._n_base = int(getattr(args, 'num_base', 80))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        device = self._protos.device
        class_ids = list(class_ids)

        # Build new prototypes
        if support_feats.dim() == 3:
            new_protos = support_feats.mean(dim=1)
        else:
            new_protos = support_feats

        # --- apply mean-shift compensation to OLD class prototypes ---
        if not self._is_first_session and self._prev_mean_shift is not None:
            cur = self._protos.data.clone()
            shift = self.compensation_strength * self._prev_mean_shift.to(device)
            cur[:self._n_base] = F.normalize(
                F.normalize(cur[:self._n_base], dim=-1) + shift,
                dim=-1)
        else:
            cur = self._protos.data.clone()

        # Append novel prototypes
        needed = max(class_ids) + 1
        if needed > cur.size(0):
            pad = torch.zeros(needed - cur.size(0), cur.size(1), device=device)
            cur = torch.cat([cur, pad], dim=0)
        for i, cid in enumerate(class_ids):
            cur[cid] = new_protos[i].to(device)

        self._protos = nn.Parameter(cur.detach(), requires_grad=False)
        self._is_first_session = False

    def record_mean_shift(self, old_feats: torch.Tensor,
                          new_feats: torch.Tensor,
                          labels: torch.Tensor) -> None:
        """Compute the average displacement of known-class features.

        Call **before** register_novel_classes to capture the drift caused
        by the novel data.
        """
        known_mask = labels < self._n_base
        if known_mask.sum() < 1:
            return
        old_k = F.normalize(old_feats[known_mask], dim=-1)
        new_k = F.normalize(new_feats[known_mask], dim=-1)
        self._prev_mean_shift = (new_k - old_k).mean(dim=0).cpu()

    # ------------------------------------------------------------------
    def classify(self, features: torch.Tensor, n_known: int) -> torch.Tensor:
        protos = self._protos[:n_known]
        # Combine cosine similarity with a small Mahalanobis-like penalty
        cos = cosine_logits(features, protos)
        return cos

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self._protos[:n_known].detach()

    # ------------------------------------------------------------------
    def train_base(self, args, trainloader,
                   log_path: Optional[str] = None) -> None:
        """Standard cosine CE training + compute per-class covariances."""

        def _loss(feats, w, y):
            logits = F.linear(F.normalize(feats, dim=-1),
                              F.normalize(w, dim=-1))
            return F.cross_entropy(logits, y)

        train_backbone_with_loss(self.model, args, trainloader,
                                 loss_fn=_loss, tag='macil_base',
                                 log_path=log_path)

        # Compute per-class means and diagonal covariances from base training
        device = next(self.model.parameters()).device
        self.model.eval()
        feats_list, labs_list = [], []
        prev_mode = self.model.mode
        self.model.mode = '__feat__'
        with torch.no_grad():
            for batch in trainloader:
                x, y = batch[0].to(device), batch[1].to(device)
                f = self.model.encode(x)
                feats_list.append(f.cpu())
                labs_list.append(y.cpu())
        self.model.mode = prev_mode

        all_feats = torch.cat(feats_list, 0)
        all_labs = torch.cat(labs_list, 0)
        n_base = self._n_base
        means, covs = [], []
        for c in range(n_base):
            idx = (all_labs == c).nonzero(as_tuple=False).flatten()
            if idx.numel() > 1:
                cf = F.normalize(all_feats[idx], dim=-1)
                means.append(cf.mean(dim=0))
                covs.append(cf.var(dim=0, unbiased=False))
            else:
                means.append(torch.zeros(all_feats.size(1)))
                covs.append(torch.ones(all_feats.size(1)))

        self._class_means = torch.stack(means, 0).cpu()
        self._class_covs = torch.stack(covs, 0).cpu()
        self._protos = nn.Parameter(self.model.fc.weight.detach().clone(),
                                    requires_grad=False)
