"""PAN — Prototype Alignment Network.

At every incremental session, novel prototypes are aligned to the
current prototype manifold through (a) EMA update on the novel side
and (b) a light-weight linear aligner trained by a contrastive
objective on the available support features so that:
  * intra-class support stays close to its prototype
  * different prototypes stay apart

Because no extra labels are required, this runs as a pure inference-time
method on the support set of each session.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CILBase, cosine_logits, train_backbone_with_loss


class PAN(CILBase):
    def __init__(self, model, args):
        super().__init__(model, args)
        dim = getattr(model, 'num_features', 512)
        self.aligner = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.aligner.weight)
        self.ema = float(getattr(args, 'pan_ema', 0.5))
        self.align_steps = int(getattr(args, 'pan_align_steps', 20))
        self.align_lr = float(getattr(args, 'pan_align_lr', 1e-2))
        self.temperature = float(getattr(args, 'pan_temperature', 16.0))
        self._protos = nn.Parameter(model.fc.weight.detach().clone(),
                                    requires_grad=False)

    # ------------------------------------------------------------------
    def _align(self, support_feats: torch.Tensor,
               labels: torch.Tensor,
               novel_ids: list) -> torch.Tensor:
        """Train the linear aligner locally then return refined novel protos."""
        device = support_feats.device
        self.aligner.to(device)
        opt = torch.optim.SGD(self.aligner.parameters(),
                              lr=self.align_lr, momentum=0.9)
        n2idx = {cid: k for k, cid in enumerate(novel_ids)}
        y = torch.tensor([n2idx[int(l)] for l in labels.tolist()],
                         device=device)
        for _ in range(self.align_steps):
            feat = self.aligner(support_feats)
            # per-class prototype on the fly
            protos = []
            for k in range(len(novel_ids)):
                mask = (y == k)
                if mask.any():
                    protos.append(feat[mask].mean(dim=0))
                else:
                    protos.append(torch.zeros(feat.size(1), device=device))
            protos = torch.stack(protos, dim=0)
            logits = cosine_logits(feat, protos, temperature=self.temperature)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            feat = self.aligner(support_feats)
            protos = torch.stack([
                feat[(y == k)].mean(dim=0) if (y == k).any()
                else torch.zeros(feat.size(1), device=device)
                for k in range(len(novel_ids))
            ], dim=0)
        return protos

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ema_merge(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return self.ema * old + (1.0 - self.ema) * new

    def register_novel_classes(self, support_feats: torch.Tensor,
                               class_ids: Iterable[int],
                               labels: torch.Tensor = None) -> None:
        device = self._protos.device
        support_feats = support_feats.to(device)
        class_ids = list(class_ids)
        if support_feats.dim() == 3:  # [n_way, n_shot, D]
            n_way, n_shot, D = support_feats.shape
            flat = support_feats.reshape(-1, D)
            lab = torch.arange(n_way, device=device).repeat_interleave(n_shot)
            lab = torch.tensor([class_ids[int(i)] for i in lab.tolist()],
                               device=device)
            new_protos = self._align(flat, lab, class_ids)
        else:
            assert labels is not None, "PAN needs per-sample labels"
            new_protos = self._align(support_feats, labels.to(device), class_ids)
        cur = self._protos.data.clone()
        needed = max(class_ids) + 1
        if needed > cur.size(0):
            pad = torch.zeros(needed - cur.size(0), cur.size(1), device=device)
            cur = torch.cat([cur, pad], dim=0)
        for i, cid in enumerate(class_ids):
            if cur[cid].abs().sum() > 0:
                cur[cid] = self._ema_merge(cur[cid], new_protos[i])
            else:
                cur[cid] = new_protos[i]
        self._protos = nn.Parameter(cur, requires_grad=False)

    # ------------------------------------------------------------------
    def classify(self, features: torch.Tensor, n_known: int) -> torch.Tensor:
        protos = self._protos[:n_known]
        return cosine_logits(features, protos, temperature=self.temperature)

    # ------------------------------------------------------------------
    # Base training: CE + contrastive regularization for aligner
    # ------------------------------------------------------------------
    def train_base(self, args, trainloader, log_path: Optional[str] = None) -> None:
        T = self.temperature
        alpha_contrast = float(getattr(args, 'pan_contrast_alpha', 0.1))
        aligner_params = list(self.aligner.parameters())

        def _loss(feats, w, y):
            logits = T * F.linear(F.normalize(feats, dim=-1),
                                  F.normalize(w, dim=-1))
            ce = F.cross_entropy(logits, y)
            # Contrastive regularization on aligned features
            aligned = self.aligner(feats)
            a_logits = T * F.linear(F.normalize(aligned, dim=-1),
                                    F.normalize(w, dim=-1))
            contrast = F.cross_entropy(a_logits, y)
            return ce + alpha_contrast * contrast

        train_backbone_with_loss(self.model, args, trainloader,
                                 loss_fn=_loss,
                                 extra_params=aligner_params,
                                 tag='pan_base',
                                 log_path=log_path)
        self._protos = nn.Parameter(self.model.fc.weight.detach().clone(),
                                    requires_grad=False)
