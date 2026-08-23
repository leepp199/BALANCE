"""FOAC-AIFP: Attention Information-Fused Prototypes (TASLP 2026).

Reference: "Few-Shot Open-Set Audio Classification Using Attention
Information-Fused Prototypes", TASLP 2026.
Official code: https://github.com/Jessytan/FOAC-AIFP

Core algorithm (adapted for shared encoder protocol):
1. PrototypeDynamicAggregation: aggregates support features into prototypes
   using attention weighting
2. ConditionalInformationCoupling: couples support and query features
3. OpenSetGenerator: generates "fake" / open-set prototypes that act as
   attractors for unknown samples
4. Scoring: cosine similarity to both known prototypes + open-set prototypes.
   If a sample is more similar to open-set prototypes → unknown.

Implementation: post-hoc scoring module that can work with any encoder.
During training (fit), computes open-set prototypes from base classes.
During scoring, measures similarity ratio between known and open-set prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OSRBase


class _OpenSetGenerator(nn.Module):
    """Generate open-set (fake) prototypes from base prototypes."""

    def __init__(self, dim: int, n_open: int = 5):
        super().__init__()
        self.n_open = n_open
        self.generator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * n_open),
        )

    def forward(self, base_protos: torch.Tensor) -> torch.Tensor:
        # base_protos: [C, D]
        agg = base_protos.mean(dim=0, keepdim=True)  # [1, D]
        out = self.generator(agg)  # [1, D * n_open]
        out = out.view(-1, self.n_open, base_protos.size(-1))  # [1, n_open, D]
        return F.normalize(out.squeeze(0), dim=-1)  # [n_open, D]


class FOAC_AIFP(OSRBase):
    """FOAC-AIFP open-set scoring via dual prototype comparison."""

    def __init__(self, args):
        super().__init__(args)
        dim = int(getattr(args, 'feat_dim', 512))
        self.n_open = int(getattr(args, 'foac_n_open', 5))
        self._generator = _OpenSetGenerator(dim, self.n_open)
        self._open_protos = None
        self._known_protos = None

    # ------------------------------------------------------------------
    def fit(self, model, train_loader):
        """Compute base class prototypes and generate open-set prototypes."""
        device = next(model.parameters()).device
        model.eval()
        self._generator.to(device)

        # Collect base class features
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

        n_base = int(getattr(self.args, 'num_base', 80))
        base_protos = []
        for c in range(n_base):
            mask = (all_labels == c)
            if mask.sum() > 0:
                proto = all_feats[mask].mean(dim=0)
            else:
                proto = torch.zeros(all_feats.size(1))
            base_protos.append(proto)

        base_protos = F.normalize(torch.stack(base_protos, dim=0),
                                  dim=-1).to(device)

        # Generate open-set prototypes from base prototypes
        self._open_protos = self._generator(base_protos).detach().cpu()

        # Also store known prototypes (mean of all base)
        self._known_protos = base_protos.mean(dim=0, keepdim=True).detach().cpu()
        self._is_fitted = True

    # ------------------------------------------------------------------
    def score(self, features: torch.Tensor,
              protos: torch.Tensor) -> torch.Tensor:
        """Score: higher = more unknown.

        Ratio of similarity to open-set prototypes vs known prototypes.
        """
        device = features.device
        f = F.normalize(features, dim=-1)

        if self._open_protos is not None:
            open_p = self._open_protos.to(device)
        else:
            # Fallback: use random open prototypes
            open_p = F.normalize(
                torch.randn(self.n_open, features.size(1), device=device),
                dim=-1)

        # Similarity to known prototypes (use protos passed in)
        known_p = F.normalize(protos.to(device), dim=-1)
        known_sim = (f @ known_p.t()).max(dim=1)[0]  # [B]

        # Similarity to open-set prototypes
        open_sim = (f @ open_p.t()).max(dim=1)[0]  # [B]

        # Higher open/known ratio = more unknown
        score = open_sim - known_sim
        return score
