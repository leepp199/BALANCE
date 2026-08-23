"""UTL: Unknown Text Learning for CLIP-based FSOR (ICCV 2025).

Reference: Ma et al., "Unknown Text Learning for CLIP-based Few-Shot Open-set
Recognition", ICCV 2025.

Core algorithm (adapted from CLIP text space to audio feature space):
1. U^2WO (Universal Unknown Word Optimization):
   - In CLIP: learns basis vectors in text embedding space to form "unknown words"
   - In audio: learn basis vectors in audio feature space to represent "unknown"
     feature directions. Unknown prototypes = linear combination of basis vectors.

2. ULS (Unknown Label Smoothing):
   - Contrastive learning between unknown prototypes and known class features
   - Unknown class labels are smoothed to a small constant, making unknown
     prototypes non-matching with known visual samples.

3. Additional known-class context to mitigate optimization conflicts.

Adaptation for prototype-based audio classifiers:
- Learn K "unknown prototypes" in the feature space that capture unknown regions
- During scoring: samples closer to unknown than known = unknown
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OSRBase


class _UnknownProtoLearner(nn.Module):
    """Learn basis vectors for representing unknown prototypes.

    U^2WO adaptation: instead of text basis vectors, we learn feature-space
    basis vectors. Unknown prototypes = linear combination of bases.
    """

    def __init__(self, dim: int, n_basis: int = 8, n_unknown: int = 3):
        super().__init__()
        self.n_basis = n_basis
        self.n_unknown = n_unknown

        # Basis vectors [n_basis, D]
        self.basis = nn.Parameter(
            F.normalize(torch.randn(n_basis, dim), dim=-1))

        # Linear combination weights [n_unknown, n_basis]
        self.combiner = nn.Linear(n_basis, n_unknown, bias=False)

    def forward(self) -> torch.Tensor:
        """Return unknown prototypes: [n_unknown, D]."""
        # Combine basis vectors to form unknown prototypes
        weights = F.softmax(self.combiner.weight, dim=1)  # [n_unknown, n_basis]
        unknowns = weights @ self.basis  # [n_unknown, D]
        return F.normalize(unknowns, dim=-1)


class UTL(OSRBase):
    """Unknown Text Learning adapted for prototype audio classifiers.

    Higher score = more likely unknown.
    """

    def __init__(self, args):
        super().__init__(args)
        dim = int(getattr(args, 'feat_dim', 512))
        self.n_basis = int(getattr(args, 'utl_n_basis', 8))
        self.n_unknown = int(getattr(args, 'utl_n_unknown', 3))
        self.uls_label = float(getattr(args, 'utl_uls_label', 0.01))

        self._learner = _UnknownProtoLearner(dim, self.n_basis, self.n_unknown)
        self._known_context = nn.Parameter(torch.zeros(1, dim))
        self._optimizer = None

    # ------------------------------------------------------------------
    def fit(self, model, train_loader):
        """Fit unknown prototypes via contrastive learning with ULS.

        Uses Unknown Label Smoothing: unknown prototypes are trained to have
        low similarity to known class features (label smoothed to small const).
        """
        device = next(model.parameters()).device
        self._learner.to(device)
        self._known_context.data = self._known_context.to(device)

        # Extract features
        model.eval()
        all_feats = []
        prev_mode = getattr(model, 'mode', None)
        if hasattr(model, 'mode'):
            model.mode = '__feat__'

        with torch.no_grad():
            for batch in train_loader:
                x = batch[0].to(device)
                f = model.encode(x).cpu()
                all_feats.append(f)

        if prev_mode is not None:
            model.mode = prev_mode

        all_feats = torch.cat(all_feats, dim=0)  # [N, D]

        # Optimize unknown prototypes
        params = list(self._learner.parameters()) + [self._known_context]
        self._optimizer = torch.optim.Adam(params, lr=0.001)

        n_iter = int(getattr(self.args, 'utl_iters', 200))
        batch_size = min(128, all_feats.size(0))
        n_known = all_feats.size(0)

        for it in range(n_iter):
            self._learner.train()

            # Sample batch
            idx = torch.randperm(n_known)[:batch_size]
            batch_feats = all_feats[idx].to(device)
            batch_feats = F.normalize(batch_feats, dim=-1)

            # Get unknown prototypes [n_unknown, D]
            unknown_protos = self._learner()

            # ULS: contrastive learning between unknown and known
            # Known should HAVE LOW similarity to unknown (ULS label = small constant)
            # Combined known features = batch_feats + known_context
            known_ctx = F.normalize(self._known_context, dim=-1)
            combined_known = F.normalize(
                batch_feats + 0.1 * known_ctx.expand_as(batch_feats), dim=-1)

            # Similarity between known features and unknown prototypes
            sim = combined_known @ unknown_protos.t()  # [B, n_unknown]
            # ULS: target label = small constant (e.g., 0.01)
            target = torch.full_like(sim, self.uls_label)
            loss_uls = F.mse_loss(torch.sigmoid(sim), target)

            # Diversity loss: unknown prototypes should be diverse
            proto_sim = unknown_protos @ unknown_protos.t()  # [n_unknown, n_unknown]
            diversity_loss = (proto_sim - torch.eye(
                self.n_unknown, device=device)).pow(2).mean()

            total_loss = loss_uls + 0.1 * diversity_loss

            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

        self._learner.eval()
        with torch.no_grad():
            self._unknown_protos = self._learner().detach().cpu()
        self._is_fitted = True

    # ------------------------------------------------------------------
    def score(self, features: torch.Tensor,
              protos: torch.Tensor) -> torch.Tensor:
        """Score: higher = more unknown.

        Measures ratio of similarity to unknown vs known prototypes.
        """
        device = features.device
        f = F.normalize(features, dim=-1)

        # Similarity to known prototypes
        known_p = F.normalize(protos.to(device), dim=-1)
        known_sim = (f @ known_p.t()).max(dim=1)[0]  # [B]

        # Similarity to unknown prototypes (learned)
        if hasattr(self, '_unknown_protos') and self._unknown_protos is not None:
            unknown_p = F.normalize(self._unknown_protos.to(device), dim=-1)
        else:
            # Fallback: random
            unknown_p = F.normalize(
                torch.randn(3, features.size(1), device=device), dim=-1)
        unknown_sim = (f @ unknown_p.t()).max(dim=1)[0]  # [B]

        # Higher unknown/known ratio = more unknown
        score = unknown_sim - known_sim
        return score
