"""COSTARR: Consolidated Open Set Technique with Attenuation (ICCV 2025).

Reference: Rabinowitz et al., "COSTARR: Consolidated Open Set Technique with
Attenuation for Robust Recognition", ICCV 2025.

Core algorithm (based on the paper's Attenuation Hypothesis):
1. Pre-attenuation: features F(x) from the encoder
2. Post-attenuation: Hadamard product H_j = F(x) ⊙ W_j (element-wise
   multiplication with the class j classifier weights)
3. For each class j, compute mean of pre-attenuation and post-attenuation
   features from the training data
4. COSTARR similarity for class j:
   C_j(x) = [cos(F(x), mu_F_j) + cos(H_j(x), mu_H_j)] / 2
5. Scale by normalized logit lambda_m = (logit_j - mean_logits) / std_logits
6. Final score: S(x) = max_j [lambda_m * C_j(x)]

Implementation for post-hoc (no re-training) use with prototype-based
classifiers: we use the prototype bank as W (classifier weights).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OSRBase


class COSTARR(OSRBase):
    """COSTARR open-set detector.

    Higher score = more likely unknown.
    """

    def __init__(self, args):
        super().__init__(args)
        # Pre-computed per-class pre-attenuation means [n_class, D]
        self._pre_means = None
        # Pre-computed per-class post-attenuation means [n_class, D]
        self._post_means = None

    # ------------------------------------------------------------------
    def fit(self, model, train_loader):
        """Compute per-class pre- and post-attenuation feature means.

        Runs the training data through the encoder (pre-attenuation)
        and the Hadamard product with fc weights (post-attenuation).
        """
        device = next(model.parameters()).device
        model.eval()

        # Collect features and labels
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

        all_feats = torch.cat(all_feats, dim=0)  # [N, D]
        all_labels = torch.cat(all_labels, dim=0)

        # Get classifier weights (used as W for Hadamard product)
        w = F.normalize(model.fc.weight.detach(), dim=-1).to(device)  # [C, D]
        n_classes = w.size(0)

        # For each class, compute pre and post means
        pre_means = []
        post_means = []
        for c in range(n_classes):
            mask = (all_labels == c)
            if mask.sum() == 0:
                pre_means.append(torch.zeros(all_feats.size(1)))
                post_means.append(torch.zeros(all_feats.size(1)))
                continue

            f_c = all_feats[mask]  # [N_c, D]
            f_c = F.normalize(f_c, dim=-1)
            # Pre-attenuation: just the features
            pre_mu = f_c.mean(dim=0)

            # Post-attenuation: Hadamard product with class weight
            w_c = F.normalize(w[c:c+1], dim=-1).to(f_c.device)
            h_c = f_c * w_c  # [N_c, D]
            post_mu = h_c.mean(dim=0)

            pre_means.append(pre_mu)
            post_means.append(post_mu)

        self._pre_means = torch.stack(pre_means, dim=0)    # [C, D]
        self._post_means = torch.stack(post_means, dim=0)  # [C, D]
        self._is_fitted = True

    # ------------------------------------------------------------------
    def score(self, features: torch.Tensor,
              protos: torch.Tensor) -> torch.Tensor:
        """COSTARR score: higher = more unknown.

        Parameters
        ----------
        features: [B, D] test sample features
        protos: [n_known, D] prototype bank (used as W)
        """
        device = features.device
        f = F.normalize(features, dim=-1)          # [B, D] pre-attenuation
        w = F.normalize(protos.to(device), dim=-1)  # [C, D]

        n_known = protos.size(0)

        # Compute per-class COSTARR similarity
        # For each class j:
        #   H_j = f ⊙ w_j  (post-attenuation)
        #   pre_sim = cos(f, pre_mu_j)
        #   post_sim = cos(h_j, post_mu_j)
        #   C_j = (pre_sim + post_sim) / 2

        # Get means (use estimated ones if fitted, else use protos as proxy)
        if self._is_fitted and self._pre_means is not None:
            pre_mu = self._pre_means[:n_known].to(device)
            post_mu = self._post_means[:n_known].to(device)
        else:
            # Fallback: use protos as pre-mean, protos*w as post-mean
            pre_mu = w
            post_mu = F.normalize(w * w, dim=-1)

        # Compute cosine similarities
        # pre_sim: [B, C]
        pre_sim = f @ pre_mu.t()  # cos similarity (already normalized)

        # post_sim for each class j
        # Expand: f [B,1,D] * w [1,C,D] -> h [B,C,D]
        h = f.unsqueeze(1) * w.unsqueeze(0)  # [B, C, D]
        h = F.normalize(h, dim=-1)
        post_sim = (h * post_mu.unsqueeze(0)).sum(dim=-1)  # [B, C]

        # Combined COSTARR similarity
        c_j = (pre_sim + post_sim) / 2.0  # [B, C]

        # Logits for normalization
        logits = f @ w.t()  # [B, C]

        # Normalized logit (lambda_m in paper)
        logit_mean = logits.mean(dim=-1, keepdim=True)
        logit_std = logits.std(dim=-1, keepdim=True) + 1e-8
        lambda_m = (logits - logit_mean) / logit_std  # [B, C]

        # Final score: max over classes of lambda_m * C_j
        score_per_class = lambda_m * c_j  # [B, C]
        final_score, _ = score_per_class.max(dim=-1)  # [B]

        # Invert: lower combined score = more unknown
        return -final_score
