"""Task 3.3 — dual-prototype heatmaps & attention maps.

Visualizes (a) the ``AttnClassifier`` positive/negative prototypes as a
heatmap and (b) the multi-head attention scores when a batch of support
features is fed through the dual-prototype attention module.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from network import MYNET, replace_base_fc  # noqa: E402
from data.dataloader import get_pretrain_dataloader, get_dataloader  # noqa: E402
from scripts.run_all_baselines import encode_loader, load_args  # noqa: E402


def heat(ax, mat, title):
    im = ax.imshow(mat, aspect='auto', cmap='magma')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_xlabel('dim'); ax.set_ylabel('class')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yml')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--out_dir', default='save/figures/attn_proto')
    p.add_argument('--gpu', default='0')
    a = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.makedirs(a.out_dir, exist_ok=True)

    args = load_args(a.config)
    from train_unopenset import set_up_datasets
    set_up_datasets(args)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MYNET(args, mode='encoder').to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu').get('params'),
                          strict=False)
    trainset, _ = get_pretrain_dataloader(args)
    model = replace_base_fc(args, trainset, model)

    cls = getattr(model, 'cls_classifier', None)
    if cls is None:
        print('no cls_classifier — abort'); return

    pos = None; neg = None
    for attr in ('pos_proto', 'positive_prototype', 'fc_pos', 'fc'):
        if hasattr(cls, attr):
            pos = getattr(cls, attr)
            pos = pos.weight if hasattr(pos, 'weight') else pos
            break
    for attr in ('neg_proto', 'negative_prototype', 'fc_neg'):
        if hasattr(cls, attr):
            neg = getattr(cls, attr)
            neg = neg.weight if hasattr(neg, 'weight') else neg
            break

    pos_mat = pos.detach().cpu().numpy() if pos is not None else None
    neg_mat = neg.detach().cpu().numpy() if neg is not None else None

    ncols = int(pos_mat is not None) + int(neg_mat is not None)
    if ncols == 0:
        print('dual prototypes not found on classifier, using model.fc only')
        pos_mat = model.fc.weight.detach().cpu().numpy()
        ncols = 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
    axes = [axes] if ncols == 1 else axes
    i = 0
    if pos_mat is not None:
        heat(axes[i], pos_mat, 'positive prototype'); i += 1
    if neg_mat is not None:
        heat(axes[i], neg_mat, 'negative prototype')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'dual_proto_heatmap.png'), dpi=160)
    plt.close(fig)

    # attention map on a support batch
    loader = get_dataloader(args, a.session)[1]
    feats, labs = encode_loader(model, loader, dev)
    feats = feats.to(dev)
    attn_module = getattr(cls, 'attn', None) or getattr(cls, 'multihead_attn', None)
    if attn_module is None:
        print('no attention module found — skip attn map')
        return
    try:
        with torch.no_grad():
            q = feats[:64].unsqueeze(0)
            out = attn_module(q, q, q) if callable(attn_module) else None
        if isinstance(out, tuple):
            attn = out[1] if out[1] is not None else out[0]
            attn = attn.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(attn[0] if attn.ndim == 3 else attn, cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title('Self-attention weights on support features')
            fig.tight_layout()
            fig.savefig(os.path.join(a.out_dir, 'attention_map.png'), dpi=160)
            plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f'attention viz failed: {e}')


if __name__ == '__main__':
    main()
