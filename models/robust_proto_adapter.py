import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustPrototypeAdapter(nn.Module):
    """DeepSets-style robust prototype weighting for noisy pseudo-clusters."""
    def __init__(self, dim=512, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1))

    def forward(self, features):
        # features: [B, K, D] or [K, D]
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        normalized = F.normalize(features, dim=-1)
        mean = F.normalize(features.mean(1, keepdim=True), dim=-1)
        expanded_mean = mean.expand_as(normalized)
        cosine = (normalized * expanded_mean).sum(-1, keepdim=True)
        scores = self.scorer(torch.cat([normalized, expanded_mean, cosine], dim=-1)).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        prototype = (weights.unsqueeze(-1) * features).sum(1)
        return prototype.squeeze(0) if squeeze else prototype
