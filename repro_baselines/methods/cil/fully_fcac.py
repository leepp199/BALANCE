"""Fully-FCAC (MAR): Multi-level Embedding + Ridge Regression (2025).

Reference: "Fully Few-Shot Class-Incremental Audio Classification Using
Multi-Level Embedding Extractor and Ridge Regression", 2025.

Core algorithm:
1. Base session: train standard encoder + fc via cross-entropy.
2. After base training, replace fc weights using Ridge Regression:
   - Extract features F from all training data.
   - Q = F^T @ Y  (cross-covariance between features and one-hot labels)
   - G = F^T @ F  (auto-covariance of features)
   - Optimise ridge parameter lambda via validation split.
   - W = solve(G + lambda*I, Q)^T  (closed-form solution)
3. Incremental sessions: accumulate Q and G, re-solve for updated W.
4. Classification via **raw dot-product** (NOT cosine) — consistent with
   the ridge-regression MSE objective.

Uses standard torchvision ResNet18 (ImageNet-pretrained) as encoder,
not MYNET. (Original paper uses AST; here ResNet18 serves as common
backbone for fair comparison across all baselines.)
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from repro_baselines.models.base_encoder import AudioResNet
from ..base import CILBase


class FullyFCAC(CILBase):
    """Ridge-regression based closed-form classifier update.

    **Key distinction**:  ``classify()`` uses **raw dot-product**
    ``features @ protos.T`` (no L2 normalisation), because the ridge
    regression objective is MSE in raw feature space.  Normalising
    would break the geometry that the closed-form solution relies on.
    """

    def __init__(self, args):
        super().__init__(args)
        n_all = int(getattr(args, 'num_all', 100))
        dim = 512

        # Own encoder — standard torchvision ResNet18
        self.model = AudioResNet(num_classes=n_all, pretrained=True,
                                 num_features=dim)

        self._protos = nn.Parameter(
            torch.zeros(n_all, dim), requires_grad=False)

        # Ridge regression accumulators
        self.register_buffer('_Q', torch.zeros(dim, n_all))
        self.register_buffer('_G', torch.zeros(dim, dim))
        self._ridge_search = getattr(args, 'ridge_search', True)
        self._ridge_range = np.logspace(-8, 8, 17)

    # ==================================================================
    def train_base(self, args, trainloader,
                   log_path: Optional[str] = None) -> None:
        """Standard CE base training, then replace fc via ridge."""
        device = next(self.model.parameters()).device
        n_base = int(getattr(args, 'num_base', 80))
        T = float(getattr(getattr(args, 'network', args), 'temperature', 16.0))

        def _loss(feats, w, y):
            logits = T * F.linear(F.normalize(feats, dim=-1),
                                  F.normalize(w, dim=-1))
            return F.cross_entropy(logits, y)

        from ..base import train_backbone_with_loss
        train_backbone_with_loss(
            self.model, args, trainloader, loss_fn=_loss,
            tag='fully_fcac_base', log_path=log_path)

        # Extract features and solve ridge regression
        self._extract_and_fit(trainloader, n_base, device, log_path)

    # ==================================================================
    def _extract_and_fit(self, loader, n_classes: int,
                         device: torch.device,
                         log_path: Optional[str] = None) -> None:
        """Extract features and solve ridge regression for fc weights."""
        self.model.eval()
        all_feats, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device), batch[1]
                f = self.model.encode(x).cpu()
                all_feats.append(f)
                all_labels.append(y)

        F_all = torch.cat(all_feats, dim=0)
        Y_all = torch.cat(all_labels, dim=0).long()

        # One-hot encoding
        Y = torch.zeros(F_all.size(0), n_classes, device=F_all.device)
        Y.scatter_(1, Y_all.unsqueeze(1), 1.0)

        # Update accumulators
        dev = self._Q.device
        self._Q[:, :n_classes] += (F_all.T @ Y).to(dev)
        self._G += (F_all.T @ F_all).to(dev)
        ridge = self._search_ridge(F_all, Y) if self._ridge_search else 1.0
        if log_path:
            with open(log_path, 'a') as fp:
                fp.write(f"[fully_fcac] ridge_lambda={ridge:.6f}\n")

        I = torch.eye(self._G.size(0), device=self._G.device)
        W = torch.linalg.solve(self._G + ridge * I, self._Q).T

        # Copy raw (un-normalised) ridge weights to prototypes
        self._protos.data[:n_classes] = W[:n_classes].to(self._protos.device)
        if hasattr(self.model, 'fc'):
            n_copy = min(self.model.fc.weight.size(0), n_classes)
            self.model.fc.weight.data[:n_copy] = W[:n_copy].to(
                self.model.fc.weight.device)

    # ==================================================================
    def _search_ridge(self, F_val: torch.Tensor,
                      Y_val: torch.Tensor) -> float:
        """Search optimal ridge parameter via 80/20 validation split."""
        n = F_val.size(0)
        n_train = int(n * 0.8)
        F_tr, F_va = F_val[:n_train], F_val[n_train:]
        Y_tr, Y_va = Y_val[:n_train], Y_val[n_train:]

        Q_val = F_tr.T @ Y_tr
        G_val = F_tr.T @ F_tr
        best_ridge = 1.0
        best_loss = float('inf')

        for ridge in self._ridge_range:
            I = torch.eye(G_val.size(0), device=G_val.device)
            W = torch.linalg.solve(G_val + ridge * I, Q_val).T
            Y_pred = F_va @ W.T
            loss = F.mse_loss(Y_pred, Y_va).item()
            if loss < best_loss:
                best_loss = loss
                best_ridge = ridge

        return best_ridge

    # ==================================================================
    @torch.no_grad()
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int],
                               labels: Optional[torch.Tensor] = None,
                               train_loader=None) -> None:
        """Register novel classes."""
        device = self._protos.device
        class_ids = list(class_ids)

        if train_loader is not None:
            self._extract_and_fit(train_loader, max(class_ids) + 1, device)
            return

        sf = support_feats.to(device)
        if sf.dim() == 3:
            new_protos = sf.mean(dim=1)
        else:
            new_protos = sf

        for i, cid in enumerate(class_ids):
            self._protos.data[cid] = new_protos[i]

        # Update Q/G with new features
        if labels is not None:
            n_known = max(class_ids) + 1
            Y_onehot = torch.zeros(sf.size(0), n_known, device=sf.device)
            Y_onehot.scatter_(1, labels.to(sf.device).unsqueeze(1), 1.0)
            self._Q += sf.T @ Y_onehot
            self._G += sf.T @ sf

    # ==================================================================
    def classify(self, features: torch.Tensor,
                 n_known: int) -> torch.Tensor:
        """Raw dot-product classification (NO cosine normalisation).

        Ridge regression solves ``min ||F W^T - Y||² + λ||W||²``; the
        resulting weights are optimal for **raw** dot-products, NOT for
        cosine similarity.  Using ``cosine_logits`` would distort the
        geometry and cause catastrophic forgetting (see debug report).
        """
        return features @ self._protos[:n_known].t()

    def prototypes(self, n_known: int) -> torch.Tensor:
        return self._protos[:n_known].detach()
