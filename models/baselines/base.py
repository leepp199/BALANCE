"""Shared interfaces for incremental and open-set baselines."""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CILBase(nn.Module):
    """Base class for class-incremental learners.

    Implementations must:
      * ``register_novel_classes(support_feats, labels)`` — insert novel class
        prototypes into ``model.fc.weight``.
      * ``classify(features)`` — return cosine-similarity logits against the
        current prototype bank (shape ``[B, C_seen]``).
    """

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

    # ------------------------------------------------------------------
    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int]) -> None:
        raise NotImplementedError

    def classify(self, features: torch.Tensor, n_known: int) -> torch.Tensor:
        raise NotImplementedError


class OSRBase:
    """Base class for open-set scorers.

    ``score(features, protos)`` returns a 1-D tensor whose larger value
    indicates *more likely to be unknown*.
    """

    def __init__(self, args):
        self.args = args

    # ------------------------------------------------------------------
    def score(self, features: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def detect(self, features: torch.Tensor, protos: torch.Tensor,
               quantile: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(unknown_mask, scores)`` using a simple batch-adaptive threshold."""
        s = self.score(features, protos)
        thr = torch.quantile(s, quantile)
        return (s >= thr), s


# ----------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------

def cosine_logits(features: torch.Tensor, protos: torch.Tensor,
                  temperature: float = 1.0) -> torch.Tensor:
    f = F.normalize(features, dim=-1)
    p = F.normalize(protos, dim=-1)
    return temperature * f @ p.t()


def build_prototype_from_support(support_feats: torch.Tensor,
                                 labels: torch.Tensor,
                                 class_ids: List[int]) -> torch.Tensor:
    protos = []
    for c in class_ids:
        idx = (labels == c).nonzero(as_tuple=False).flatten()
        protos.append(support_feats[idx].mean(dim=0))
    return torch.stack(protos, dim=0)


# ----------------------------------------------------------------------
# Shared base-train scaffold for CIL baselines (CEC / AMFO / PAN)
# ----------------------------------------------------------------------

def _extract_features(model, x: torch.Tensor) -> torch.Tensor:
    """Run encoder.encode but bypass the fc head by toggling mode."""
    prev_mode = model.mode
    model.mode = '__feat__'
    try:
        f = model.encode(x)
    finally:
        model.mode = prev_mode
    return f


def train_backbone_with_loss(model,
                             args,
                             trainloader,
                             loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                             extra_params: Optional[Iterable[nn.Parameter]] = None,
                             epochs: Optional[int] = None,
                             tag: str = 'baseline',
                             log_path: Optional[str] = None) -> None:
    """Standard supervised base training driver shared by CIL baselines.

    Parameters
    ----------
    loss_fn(features, fc_weight, labels) -> scalar loss
        method-specific loss using extracted 512-d features and the cosine
        classifier weights ``model.fc.weight[:num_base]``.
    extra_params : iterable of additional learnable parameters (e.g. PAN aligner)
    """
    device = next(model.parameters()).device
    n_base = int(getattr(args, 'num_base', 80))
    epochs = int(epochs if epochs is not None else getattr(args.epochs, 'epochs_std', 30))

    params = list(model.encoder.parameters()) + [model.fc.weight] + ([model.fc.bias] if model.fc.bias is not None else [])
    if extra_params is not None:
        for p in extra_params:
            params.append(p)
    optimizer = torch.optim.SGD([p for p in params if p.requires_grad],
                                lr=float(getattr(args, 'learning_rate', 0.05)),
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    log_fp = open(log_path, 'a') if log_path else None
    if log_fp:
        log_fp.write(f"=== train_backbone_with_loss tag={tag} epochs={epochs} num_base={n_base} ===\n")

    for ep in range(epochs):
        model.train()
        loss_meter = 0.0
        n_seen = 0
        n_correct = 0
        for batch in trainloader:
            x, y = batch[0].to(device), batch[1].to(device)
            y = y.long()
            feats = _extract_features(model, x)
            w = model.fc.weight[:n_base]
            loss = loss_fn(feats, w, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                logits = F.linear(F.normalize(feats, dim=-1), F.normalize(w, dim=-1))
                pred = logits.argmax(dim=-1)
                n_correct += int((pred == y).sum().item())
                n_seen += y.size(0)
            loss_meter += float(loss.item()) * y.size(0)
        scheduler.step()
        acc = n_correct / max(n_seen, 1)
        avg = loss_meter / max(n_seen, 1)
        msg = f"[{tag}] epoch {ep+1}/{epochs} loss={avg:.4f} acc={acc:.4f}"
        print(msg)
        if log_fp:
            log_fp.write(msg + "\n")
            log_fp.flush()
    if log_fp:
        log_fp.close()
