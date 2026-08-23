"""OpenMax OSR detector.

Fits Weibull distributions to the distances between features and their
class centroids, then recalibrates softmax scores to include an explicit
"unknown" probability.

Reference: Bendale & Boult, "Towards Open Set Deep Networks", CVPR 2016.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class OpenMaxOSR:
    """OpenMax-based open-set recognition scorer.

    score = probability of "unknown" class.
    Higher score → more likely unknown.
    """

    def __init__(self, tailsize: int = 20, alpha: int = 3,
                 distance_type: str = 'euclidean'):
        self.tailsize = tailsize
        self.alpha = alpha  # number of top classes to revise
        self.distance_type = distance_type
        self.weibull_params = {}  # {class_id: (scale, shape, translate)}
        self.threshold = None
        self.num_classes = 0

    @staticmethod
    def _fit_weibull_tail(distances: np.ndarray, tailsize: int):
        """Fit Weibull to the tail of distances.

        Uses simplified MLE: sort distances, take largest tailsize,
        fit 2-parameter Weibull using method of moments approximation.

        Returns (scale, shape, translate).
        """
        dists = np.sort(distances)
        if len(dists) < tailsize:
            tailsize = max(2, len(dists) // 2)
        tail = dists[-tailsize:]
        translate = dists[-tailsize] if tailsize > 0 else 0.0
        shifted = tail - translate + 1e-8
        # Method of moments for Weibull
        log_x = np.log(shifted)
        mean_log = np.mean(log_x)
        var_log = np.var(log_x) if len(log_x) > 1 else 0.1
        # shape (k) approximation
        shape = 1.0 / max(np.sqrt(6.0 * var_log / np.pi**2), 0.01)
        # scale (lambda) approximation
        euler = 0.5772
        scale = np.exp(mean_log + euler / shape)
        return float(scale), float(shape), float(translate)

    @staticmethod
    def _weibull_cdf(x: float, scale: float, shape: float,
                      translate: float) -> float:
        """Weibull CDF at x."""
        if x <= translate:
            return 0.0
        z = (x - translate) / max(scale, 1e-8)
        return 1.0 - float(np.exp(-(z ** shape)))

    def calibrate(self, features: torch.Tensor, labels: torch.Tensor,
                  known_pct: float = 0.95) -> float:
        """Fit Weibull models per known class.

        Args:
            features: [N, D] known-class features.
            labels: [N] class indices.
            known_pct: percentile for threshold.

        Returns:
            threshold value.
        """
        # Move to CPU for numpy operations
        feats_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        self.num_classes = int(labels_np.max()) + 1

        # Compute class centroids
        centroids = []
        for c in range(self.num_classes):
            mask = labels_np == c
            if mask.sum() > 0:
                centroids.append(feats_np[mask].mean(axis=0))
            else:
                centroids.append(np.zeros(feats_np.shape[1]))
        centroids = np.stack(centroids, axis=0)  # [C, D]

        # Fit Weibull per class
        for c in range(self.num_classes):
            mask = labels_np == c
            if mask.sum() < 2:
                self.weibull_params[c] = (1.0, 1.0, 0.0)
                continue
            cls_feats = feats_np[mask]
            # Distance to this class centroid
            if self.distance_type == 'euclidean':
                dists = np.linalg.norm(cls_feats - centroids[c], axis=1)
            else:  # cosine distance
                norm_c = centroids[c] / (np.linalg.norm(centroids[c]) + 1e-8)
                norm_f = cls_feats / (np.linalg.norm(cls_feats, axis=1, keepdims=True) + 1e-8)
                dists = 1.0 - (norm_f @ norm_c)
            scale, shape, translate = self._fit_weibull_tail(dists, self.tailsize)
            self.weibull_params[c] = (scale, shape, translate)

        # Calibrate threshold
        scores = self.score(torch.from_numpy(feats_np),
                           torch.from_numpy(centroids))
        self.threshold = float(torch.quantile(scores, known_pct))
        return self.threshold

    def score(self, features: torch.Tensor,
              prototypes: torch.Tensor) -> torch.Tensor:
        """Compute OpenMax unknown probability.

        Higher score → more likely unknown.
        """
        feats_np = features.cpu().numpy()
        protos_np = prototypes.cpu().numpy()
        N = feats_np.shape[0]
        C = min(protos_np.shape[0], self.num_classes) if self.num_classes > 0 else protos_np.shape[0]

        unknown_probs = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if self.distance_type == 'euclidean':
                dists = np.linalg.norm(feats_np[i] - protos_np[:C], axis=1)
            else:
                norm_f = feats_np[i] / (np.linalg.norm(feats_np[i]) + 1e-8)
                norm_p = protos_np[:C] / (np.linalg.norm(protos_np[:C], axis=1, keepdims=True) + 1e-8)
                dists = 1.0 - (norm_p @ norm_f)

            # Convert distances to Weibull CDF probabilities
            w_scores = np.ones(C, dtype=np.float32)
            for c in range(C):
                if c in self.weibull_params:
                    s, k, t = self.weibull_params[c]
                    w_scores[c] = self._weibull_cdf(float(dists[c]), s, k, t)

            # Sort activations descending by confidence (1 - w_score)
            sorted_idx = np.argsort(1.0 - w_scores)
            alpha = min(self.alpha, C)

            # Recalibrate top-alpha scores (reduce them)
            recalib = np.zeros(C + 1, dtype=np.float32)
            for j in range(C):
                recalib[j] = 1.0 - w_scores[j]  # confidence = 1 - outlier-prob

            for j in range(alpha):
                idx = sorted_idx[-(j + 1)]
                recalib[idx] *= (1.0 - w_scores[idx])

            # Unknown score = 1 - sum(recalibrated)
            recalib[-1] = max(0.0, 1.0 - recalib[:C].sum())
            unknown_probs[i] = recalib[-1]

        return torch.from_numpy(unknown_probs).to(features.device)

    def is_known(self, features: torch.Tensor,
                 prototypes: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True = known."""
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before is_known()")
        return self.score(features, prototypes) < self.threshold

    def get_scores_for_auroc(self, features: torch.Tensor,
                              prototypes: torch.Tensor) -> torch.Tensor:
        """Return scores where HIGHER = more likely KNOWN (for AUROC)."""
        return 1.0 - self.score(features, prototypes)
