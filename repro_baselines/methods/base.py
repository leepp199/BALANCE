"""Shared interfaces and helpers for CIL and OSR baseline methods.

CIL methods now OWN their own encoder model (no more shared MYNET).
Each subclass creates ``self.model`` — a standard AudioResNet or
method-specific architecture — inside ``__init__``.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CIL Base
# ======================================================================

class CILBase(nn.Module):
    """Base class for class-incremental learners.

    Each subclass creates **its own encoder** in ``__init__`` and
    stores it in ``self.model``.

    Subclasses must implement:
      * ``train_base(args, trainloader, log_path)``
      * ``register_novel_classes(support_feats, class_ids)``
      * ``classify(features, n_known)``
      * ``prototypes(n_known)``
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = None       # Subclasses create their own encoder

    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        raise NotImplementedError

    def classify(self, features: torch.Tensor, n_known: int) -> torch.Tensor:
        raise NotImplementedError

    def train_base(self, args, trainloader,
                   log_path: Optional[str] = None) -> None:
        raise NotImplementedError

    def prototypes(self, n_known: int) -> torch.Tensor:
        """Return the current prototype bank (first n_known rows)."""
        raise NotImplementedError


# ======================================================================
# OSR Base
# ======================================================================

class OSRBase:
    """Base class for open-set scorers.

    ``score(features, protos)`` returns a 1-D tensor whose larger value
    indicates *more likely to be unknown*.
    """

    def __init__(self, args):
        self.args = args
        self._is_fitted = False

    def score(self, features: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, model, train_loader) -> None:
        """Optional fitting step (e.g. compute class statistics)."""
        self._is_fitted = True

    def detect(self, features: torch.Tensor, protos: torch.Tensor,
               quantile: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(unknown_mask, scores)`` using a batch-adaptive threshold."""
        s = self.score(features, protos)
        thr = torch.quantile(s, quantile)
        return (s >= thr), s


# ======================================================================
# Common helpers
# ======================================================================

def cosine_logits(features: torch.Tensor, protos: torch.Tensor,
                  temperature: float = 1.0) -> torch.Tensor:
    """Cosine-similarity logits: ``temperature * (f @ p.T)`` after L2 norm."""
    f = F.normalize(features, dim=-1)
    p = F.normalize(protos, dim=-1)
    return temperature * (f @ p.t())


def build_prototype_from_support(support_feats: torch.Tensor,
                                 labels: torch.Tensor,
                                 class_ids: List[int]) -> torch.Tensor:
    """Build per-class prototypes by averaging support features."""
    protos = []
    for c in class_ids:
        idx = (labels == c).nonzero(as_tuple=False).flatten()
        protos.append(support_feats[idx].mean(dim=0))
    return torch.stack(protos, dim=0)


# ======================================================================
# Shared base-train driver
# ======================================================================

def _get_lr(args, default=0.005):
    """Extract learning rate from nested or flat config."""
    if hasattr(args, 'lr'):
        lr = args.lr
        if hasattr(lr, 'lr_std'):
            return float(lr.lr_std)
        if isinstance(lr, dict):
            return float(lr.get('lr_std', default))
    return float(getattr(args, 'learning_rate', default))


def train_backbone_with_loss(model, args, trainloader,
                             loss_fn=None,
                             extra_params: Optional[Iterable[nn.Parameter]] = None,
                             epochs: Optional[int] = None,
                             tag: str = 'baseline',
                             log_path: Optional[str] = None) -> None:
    """Standard supervised base training shared by CIL baselines.

    The ``model`` must expose:
      - ``model.encoder`` or have a forward that handles ``(x, labels)``
      - ``model.fc.weight`` for the classifier head
      - ``model.encode(x)`` for feature extraction

    Parameters
    ----------
    loss_fn(features, fc_weight, labels) -> scalar loss
    extra_params : additional learnable parameters beyond encoder+fc
    """
    device = next(model.parameters()).device
    n_base = int(getattr(args, 'num_base', 80))
    epochs = int(epochs if epochs is not None
                else getattr(args.epochs, 'epochs_std', 30))

    # Collect trainable params
    params = []
    if hasattr(model, 'encoder') and model.encoder is not None:
        params.extend(model.encoder.parameters())
    else:
        # Fallback: all params except fc
        for name, p in model.named_parameters():
            if 'fc' not in name and p.requires_grad:
                params.append(p)

    params.append(model.fc.weight)
    if hasattr(model.fc, 'bias') and model.fc.bias is not None:
        params.append(model.fc.bias)

    if extra_params is not None:
        for p in extra_params:
            params.append(p)

    optimizer = torch.optim.SGD(
        [p for p in params if p.requires_grad],
        lr=_get_lr(args, default=0.005),
        momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1))

    log_fp = open(log_path, 'a') if log_path else None
    if log_fp:
        log_fp.write(f"=== train_backbone tag={tag} epochs={epochs} "
                     f"n_base={n_base} ===\n")

    for ep in range(epochs):
        model.train()
        loss_meter = 0.0
        n_seen = 0
        n_correct = 0

        for batch in trainloader:
            x, y = batch[0].to(device), batch[1].to(device)
            y = y.long()

            feats = model.encode(x)          # (B, D)
            w = model.fc.weight[:n_base]

            if loss_fn is not None:
                loss = loss_fn(feats, w, y)
            else:
                logits = F.linear(F.normalize(feats, dim=-1),
                                  F.normalize(w, dim=-1))
                loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                logits = F.linear(F.normalize(feats, dim=-1),
                                  F.normalize(w, dim=-1))
                pred = logits.argmax(dim=-1)
                n_correct += int((pred == y).sum().item())
                n_seen += y.size(0)
            loss_meter += float(loss.item()) * y.size(0)

        scheduler.step()
        acc = n_correct / max(n_seen, 1)
        avg = loss_meter / max(n_seen, 1)
        msg = f"[{tag}] ep {ep+1}/{epochs} loss={avg:.4f} acc={acc:.4f}"
        print(msg)
        if log_fp:
            log_fp.write(msg + "\n")
            log_fp.flush()

    if log_fp:
        log_fp.close()


__all__ = [
    "CILBase", "OSRBase",
    "cosine_logits", "build_prototype_from_support",
    "train_backbone_with_loss",
]
