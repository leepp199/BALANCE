"""Deep Feature Structure Bank primitives.

The centers are shared reference points in task-adapted deep feature space.  They are
not semantic classes, acoustic units, or an acoustic vocabulary.  ``K`` is the number
of feature structures and is independent of the number of classification classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StructureOutputs:
    assignments: torch.Tensor       # [B, H, W]
    structural_response: torch.Tensor  # [B, K]
    structure_residual: torch.Tensor   # [B]


class DeepFeatureStructureBank:
    """Frozen cross-sample structure centers with cosine assignments."""

    def __init__(self, centers: torch.Tensor, temperature: float = 0.1):
        if centers.ndim != 2:
            raise ValueError(f"centers must be [K, D], got {tuple(centers.shape)}")
        if centers.numel() == 0 or not torch.isfinite(centers).all():
            raise ValueError("centers must be finite and non-empty")
        self.centers = F.normalize(centers.detach().float(), dim=-1)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    @property
    def num_clusters(self) -> int:
        return self.centers.size(0)

    @property
    def feature_dim(self) -> int:
        return self.centers.size(1)

    def to(self, device: torch.device | str) -> "DeepFeatureStructureBank":
        self.centers = self.centers.to(device)
        return self

    def compute(self, feature_map: torch.Tensor) -> StructureOutputs:
        """Compute hard targets, response distribution, and cosine residual.

        ``feature_map`` must be the unpooled layer4 map ``[B, 512, H, W]``.
        The calculation is batched and never fits K-means during forward/inference.
        """
        if feature_map.ndim != 4:
            raise ValueError(f"feature_map must be [B, D, H, W], got {tuple(feature_map.shape)}")
        batch, dim, height, width = feature_map.shape
        if dim != self.feature_dim:
            raise ValueError(f"feature dim {dim} does not match bank dim {self.feature_dim}")
        descriptors = feature_map.permute(0, 2, 3, 1).reshape(batch, height * width, dim)
        descriptors = F.normalize(descriptors.float(), dim=-1)
        centers = self.centers.to(descriptors.device)
        similarities = descriptors @ centers.t()  # [B, HW, K]
        assignments = similarities.argmax(dim=-1).view(batch, height, width)
        soft_assignments = F.softmax(similarities / self.temperature, dim=-1)
        structural_response = soft_assignments.mean(dim=1)
        structure_residual = (1.0 - similarities.max(dim=-1).values).mean(dim=1)
        return StructureOutputs(assignments, structural_response, structure_residual)

    def state_dict(self) -> Dict[str, torch.Tensor | float | int]:
        return {
            "centers": self.centers.cpu(),
            "temperature": self.temperature,
            "num_clusters": self.num_clusters,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "DeepFeatureStructureBank":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        centers = payload["centers"] if isinstance(payload, dict) else payload
        temperature = float(payload.get("temperature", 0.1)) if isinstance(payload, dict) else 0.1
        return cls(centers, temperature=temperature)


def descriptor_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, D, H, W]`` to normalized CPU descriptors ``[BHW, D]``."""
    if feature_map.ndim != 4:
        raise ValueError(f"Expected [B, D, H, W], got {tuple(feature_map.shape)}")
    descriptors = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.size(1))
    return F.normalize(descriptors.float(), dim=-1).cpu()


class MaskedStructurePredictor(nn.Module):
    """Predict frozen DFSB assignments from masked layer4 maps.

    H/W are treated only as latent spatial axes. The predictor never fits or updates
    K-means; its targets come from the unmasked map and the frozen shared bank.
    """

    def __init__(self, feature_dim: int, num_clusters: int, hidden_dim: int = 256,
                 mask_ratio: float = 0.3, mask_mode: str = "random"):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if mask_mode not in {"random", "axis_h", "axis_w", "dual_axis"}:
            raise ValueError(f"Unsupported mask mode: {mask_mode}")
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = mask_mode
        self.mask_token = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))
        self.predictor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_clusters, kernel_size=1),
        )

    def make_mask(self, batch: int, height: int, width: int, device) -> torch.Tensor:
        if self.mask_mode == "random":
            mask = torch.rand(batch, height, width, device=device) < self.mask_ratio
        elif self.mask_mode == "axis_h":
            rows = torch.rand(batch, height, 1, device=device) < self.mask_ratio
            mask = rows.expand(-1, -1, width)
        elif self.mask_mode == "axis_w":
            cols = torch.rand(batch, 1, width, device=device) < self.mask_ratio
            mask = cols.expand(-1, height, -1)
        else:
            axis_ratio = 1.0 - (1.0 - self.mask_ratio) ** 0.5
            rows = torch.rand(batch, height, 1, device=device) < axis_ratio
            cols = torch.rand(batch, 1, width, device=device) < axis_ratio
            mask = rows.expand(-1, -1, width) | cols.expand(-1, height, -1)
        # Guarantee at least one supervised location for every sample.
        empty = ~mask.flatten(1).any(1)
        if empty.any():
            chosen = torch.randint(height * width, (int(empty.sum()),), device=device)
            mask[empty] = False
            mask[empty].view(-1, height * width).scatter_(1, chosen[:, None], True)
        return mask

    def forward(self, feature_map: torch.Tensor, centers: torch.Tensor):
        batch, dim, height, width = feature_map.shape
        if dim != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {dim}")
        with torch.no_grad():
            descriptors = F.normalize(feature_map.detach(), dim=1)
            normalized_centers = F.normalize(centers.detach(), dim=-1)
            target_logits = torch.einsum("bdhw,kd->bkhw", descriptors, normalized_centers)
            targets = target_logits.argmax(dim=1)  # [B,H,W]
        mask = self.make_mask(batch, height, width, feature_map.device)
        masked_map = torch.where(mask[:, None], self.mask_token.to(feature_map.dtype), feature_map)
        logits = self.predictor(masked_map)  # [B,K,H,W]
        loss = F.cross_entropy(logits.permute(0, 2, 3, 1)[mask], targets[mask])
        with torch.no_grad():
            accuracy = (logits.argmax(1)[mask] == targets[mask]).float().mean()
        return loss, accuracy, mask, targets
