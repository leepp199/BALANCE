"""Latent Structure Reference Bank primitives.

LSRB centers are shared reference points in the task-adapted latent space. They are
neither semantic classes nor physical time/frequency units. The bank is fitted on base
descriptors, participates in base representation learning, and is frozen afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StructureOutputs:
    assignments: torch.Tensor
    structural_response: torch.Tensor
    structure_residual: torch.Tensor


class LatentStructureReferenceBank:
    """Frozen cross-sample latent reference centers with cosine assignments."""

    def __init__(self, centers: torch.Tensor, temperature: float = 0.1):
        if centers.ndim != 2:
            raise ValueError(f"centers must be [K, D], got {tuple(centers.shape)}")
        if centers.numel() == 0 or not torch.isfinite(centers).all():
            raise ValueError("centers must be finite and non-empty")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.centers = F.normalize(centers.detach().float(), dim=-1)
        self.temperature = float(temperature)

    @property
    def num_clusters(self) -> int:
        return self.centers.size(0)

    @property
    def feature_dim(self) -> int:
        return self.centers.size(1)

    def to(self, device: torch.device | str) -> "LatentStructureReferenceBank":
        self.centers = self.centers.to(device)
        return self

    def compute(self, feature_map: torch.Tensor) -> StructureOutputs:
        """Return assignments, response distributions, and residuals for [B,D,H,W]."""
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must be [B, D, H, W], got {tuple(feature_map.shape)}"
            )
        batch, dim, height, width = feature_map.shape
        if dim != self.feature_dim:
            raise ValueError(f"feature dim {dim} does not match bank dim {self.feature_dim}")
        descriptors = feature_map.permute(0, 2, 3, 1).reshape(batch, height * width, dim)
        descriptors = F.normalize(descriptors.float(), dim=-1)
        similarities = descriptors @ self.centers.to(descriptors.device).t()
        assignments = similarities.argmax(dim=-1).view(batch, height, width)
        structural_response = F.softmax(
            similarities / self.temperature, dim=-1
        ).mean(dim=1)
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
    def from_state_dict(cls, payload: Dict) -> "LatentStructureReferenceBank":
        if "centers" not in payload:
            raise KeyError("LSRB state is missing centers")
        return cls(payload["centers"], temperature=float(payload.get("temperature", 0.1)))

    @classmethod
    def load(
        cls, path: str | Path, map_location: str = "cpu"
    ) -> "LatentStructureReferenceBank":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if isinstance(payload, dict) and "structure_bank" in payload:
            payload = payload["structure_bank"]
        if isinstance(payload, dict):
            return cls.from_state_dict(payload)
        return cls(payload)


def descriptor_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert [B,D,H,W] to normalized CPU descriptors [BHW,D]."""
    if feature_map.ndim != 4:
        raise ValueError(f"expected [B, D, H, W], got {tuple(feature_map.shape)}")
    descriptors = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.size(1))
    return F.normalize(descriptors.float(), dim=-1).cpu()


class MaskedStructurePredictor(nn.Module):
    """Predict frozen LSRB assignments from masked latent feature maps."""

    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        hidden_dim: int = 256,
        mask_ratio: float = 0.3,
        mask_mode: str = "random",
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if mask_mode not in {"random", "axis_h", "axis_w", "dual_axis"}:
            raise ValueError(f"unsupported mask mode: {mask_mode}")
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
            columns = torch.rand(batch, 1, width, device=device) < self.mask_ratio
            mask = columns.expand(-1, height, -1)
        else:
            axis_ratio = 1.0 - (1.0 - self.mask_ratio) ** 0.5
            rows = torch.rand(batch, height, 1, device=device) < axis_ratio
            columns = torch.rand(batch, 1, width, device=device) < axis_ratio
            mask = rows.expand(-1, -1, width) | columns.expand(-1, height, -1)
        empty = ~mask.flatten(1).any(1)
        if empty.any():
            chosen = torch.randint(height * width, (int(empty.sum()),), device=device)
            mask[empty] = False
            mask[empty].view(-1, height * width).scatter_(1, chosen[:, None], True)
        return mask

    def forward(self, feature_map: torch.Tensor, centers: torch.Tensor):
        batch, dim, height, width = feature_map.shape
        if dim != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {dim}")
        with torch.no_grad():
            descriptors = F.normalize(feature_map.detach(), dim=1)
            normalized_centers = F.normalize(centers.detach(), dim=-1)
            targets = torch.einsum(
                "bdhw,kd->bkhw", descriptors, normalized_centers
            ).argmax(dim=1)
        mask = self.make_mask(batch, height, width, feature_map.device)
        masked_map = torch.where(mask[:, None], self.mask_token.to(feature_map.dtype), feature_map)
        logits = self.predictor(masked_map)
        loss = F.cross_entropy(logits.permute(0, 2, 3, 1)[mask], targets[mask])
        with torch.no_grad():
            accuracy = (logits.argmax(1)[mask] == targets[mask]).float().mean()
        return loss, accuracy, mask, targets
