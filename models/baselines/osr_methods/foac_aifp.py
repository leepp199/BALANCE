"""FOAC-AIFP — Attention Information-Fused Prototypes for OSR (Li et al., 2026).

Reference: "Few-Shot Open-Set Audio Classification Using Attention
Information-Fused Prototypes", IEEE TASLP 2026.

Core idea:
- After base training, compute base-class prototypes from training features.
- Use an attention-based generator (NPM) to synthesise "open prototypes"
  representing the unknown space.
- A test sample whose nearest neighbour is an open prototype is scored high
  (likely unknown).

In our baseline pipeline we do NOT run the full episodic meta-training loop.
Instead we implement the *scoring function* that FOAC-AIFP would produce:
1. Extract base-class prototypes by running the training set through the
   (frozen) encoder.
2. Generate open prototypes via a lightweight attention-based aggregator
   trained offline using the same base training features as pseudo-support.
3. OSR score = distance to nearest open prototype minus distance to
   nearest known prototype.

This keeps the implementation self-contained while preserving the key
attention-fused prototype concept.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import OSRBase


# ---------------------------------------------------------------------------
# Lightweight attention-based open prototype generator
# ---------------------------------------------------------------------------

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        w = F.softmax(attn, dim=-1)
        return torch.matmul(w, v), w


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.attn = _ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        B = q.size(0)
        q = self.wq(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        o, _ = self.attn(q, k, v)
        o = o.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.linear(o)


class _OpenSetGenerator(nn.Module):
    """Simplified NPM (negative prototype module) from FOAC-AIFP.

    Takes base-class prototypes and generates a small set of open prototypes.
    """

    def __init__(self, dim: int = 512, n_heads: int = 1):
        super().__init__()
        self.attn_known = _MultiHeadAttention(dim, n_heads)
        self.attn_open = _MultiHeadAttention(dim, n_heads)
        self.agg = nn.Sequential(
            nn.Linear(dim, dim), nn.LeakyReLU(0.5),
            nn.Dropout(0.5), nn.Linear(dim, dim),
        )

    def forward(self, known_protos: torch.Tensor,
                open_seeds: torch.Tensor) -> torch.Tensor:
        """Generate open prototypes.

        Args:
            known_protos: [n_base, D] base class prototypes.
            open_seeds: [n_open, D] seed vectors (e.g. random or cluster
                centers from unknown-augmented training data).
        Returns:
            open_protos: [n_open, D] generated open prototypes.
        """
        # Attend known protos to produce context
        ctx = self.attn_known(open_seeds.unsqueeze(0),
                               known_protos.unsqueeze(0),
                               known_protos.unsqueeze(0)).squeeze(0)
        # Attend to open seeds
        out = self.attn_open(ctx.unsqueeze(0), ctx.unsqueeze(0),
                              ctx.unsqueeze(0)).squeeze(0)
        out = self.agg(out)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# OSR scoring wrapper
# ---------------------------------------------------------------------------

class FOAC_AIFP(OSRBase):
    """Open-set scoring via attention-fused open prototypes.

    Requires no episodic meta-training — the generator is trained once on
    the base training features and then frozen.
    """

    def __init__(self, args):
        super().__init__(args)
        self.dim = getattr(args, 'feat_dim', 512)
        self.n_open = int(getattr(args, 'foac_n_open', 5))
        self._generator = _OpenSetGenerator(self.dim).to(self._get_device())
        self._known_protos = None
        self._open_protos = None
        self._trained = False

    @staticmethod
    def _get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    def fit(self, encoder, trainloader):
        """Train the open prototype generator using base training features.

        Args:
            encoder: MYNET model with .encode(x) returning 512-d features.
            trainloader: DataLoader for base training set.
        """
        device = self._get_device()
        encoder.eval()
        encoder.to(device)

        # 1. Extract all training features
        feats, labs = [], []
        with torch.no_grad():
            for batch in trainloader:
                x, y = batch[0].to(device), batch[1].to(device)
                f = encoder.encode(x)
                feats.append(f.cpu())
                labs.append(y.cpu())
        all_feats = torch.cat(feats, 0)  # [N, D]
        all_labs = torch.cat(labs, 0)

        # 2. Compute per-class prototypes (known)
        n_base = int(all_labs.max().item()) + 1
        protos = []
        for c in range(n_base):
            idx = (all_labs == c).nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                protos.append(all_feats[idx].mean(dim=0))
            else:
                protos.append(torch.zeros(self.dim))
        self._known_protos = F.normalize(torch.stack(protos, 0), dim=-1).to(device)

        # 3. Generate "open seeds" by k-means on the 5% most uncertain samples
        #    (samples whose max cosine similarity is lowest)
        with torch.no_grad():
            sims = all_feats @ self._known_protos.cpu().t()  # [N, n_base]
            max_sim, _ = sims.max(dim=-1)
            n_seed = min(int(all_feats.size(0) * 0.05), self.n_open * 20)
            uncertain_idx = max_sim.argsort()[:n_seed]
            uncertain = all_feats[uncertain_idx]

            if uncertain.size(0) >= self.n_open:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=self.n_open, n_init=5,
                            random_state=42).fit(uncertain.numpy())
                open_seeds = torch.from_numpy(km.cluster_centers_).float()
            else:
                open_seeds = uncertain[:self.n_open]

        open_seeds = F.normalize(open_seeds, dim=-1).to(device)

        # 4. Train the generator for a few steps
        known = self._known_protos.unsqueeze(0)  # [1, n_base, D]
        seeds = open_seeds.unsqueeze(0)           # [1, n_open, D]
        opt = torch.optim.Adam(self._generator.parameters(), lr=1e-3)
        self._generator.train()
        for _ in range(200):
            out = self._generator(known.squeeze(0), seeds.squeeze(0))
            # Loss: open prototypes should be dissimilar to known prototypes
            # and diverse among themselves
            sim_known = out @ known.squeeze(0).t()  # [n_open, n_base]
            loss_sim = sim_known.abs().mean()
            # diversity: cosine similarity among open prototypes
            sim_self = out @ out.t()  # [n_open, n_open]
            loss_div = -sim_self[~torch.eye(self.n_open, dtype=torch.bool,
                                             device=device)].abs().mean()
            loss = loss_sim + 0.1 * loss_div
            opt.zero_grad()
            loss.backward()
            opt.step()

        self._generator.eval()
        with torch.no_grad():
            self._open_protos = self._generator(
                self._known_protos, open_seeds)
        self._trained = True

    # ------------------------------------------------------------------
    def score(self, features: torch.Tensor,
              protos: torch.Tensor) -> torch.Tensor:
        """OSR score: higher = more likely unknown."""
        if not self._trained:
            raise RuntimeError("FOAC_AIFP scorer not fitted yet — call .fit()")
        device = features.device
        known = F.normalize(protos, dim=-1).to(device)
        open_p = self._open_protos.to(device)
        f = F.normalize(features, dim=-1)
        # Score = max sim to open - max sim to known
        sim_open = (f @ open_p.t()).max(dim=-1).values
        sim_known = (f @ known.t()).max(dim=-1).values
        return sim_open - sim_known
