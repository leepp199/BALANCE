"""Task 3.1 — feature-space visualizations.

Dumps t-SNE and UMAP (if available) projections of encoder features and
class prototypes at each session, plus a prototype-drift line plot.

Usage::

    python -m scripts.viz_feature_space --ckpt save/base_train_for_meta_ls.pth \
        --config configs/default.yml --session 1 2 3 4 \
        --out_dir save/figures/feat_space
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from network import MYNET, replace_base_fc  # noqa: E402
from data.dataloader import (  # noqa: E402
    get_dataloader,
    get_pretrain_dataloader,
    get_testloader,
)
from scripts.run_all_baselines import encode_loader, load_args  # noqa: E402


def _safe_umap(feats):
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(feats)
    except Exception:  # pragma: no cover
        return None


def scatter(ax, proj, labs, protos_proj=None, title=''):
    uniq = np.unique(labs)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(20, len(uniq))))
    for i, c in enumerate(uniq):
        m = labs == c
        ax.scatter(proj[m, 0], proj[m, 1], s=8, alpha=0.6,
                    color=cmap[i % len(cmap)], label=f'c{c}')
    if protos_proj is not None:
        ax.scatter(protos_proj[:, 0], protos_proj[:, 1], marker='X',
                    s=140, c='k', edgecolors='yellow', linewidths=1.5,
                    label='proto')
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yml')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--session', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--out_dir', default='save/figures/feat_space')
    p.add_argument('--gpu', default='0')
    a = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.makedirs(a.out_dir, exist_ok=True)

    args = load_args(a.config)
    from train_unopenset import set_up_datasets
    set_up_datasets(args)
    args.cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if args.cuda else 'cpu')

    model = MYNET(args, mode='encoder').to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu').get('params'),
                          strict=False)
    trainset, _ = get_pretrain_dataloader(args)
    model = replace_base_fc(args, trainset, model)

    proto_history = []
    for s in a.session:
        loader = get_testloader(args, s)[1] if s == 0 else get_dataloader(args, s)[1]
        feats, labs = encode_loader(model, loader, dev)
        feats, labs = feats.numpy(), labs.numpy()
        n_seen = args.num_base + max(s, 0) * args.way
        protos = model.fc.weight[:n_seen, :].detach().cpu().numpy()
        proto_history.append(protos.copy())

        all_pts = np.concatenate([feats, protos], 0)
        tsne = TSNE(n_components=2, perplexity=30, init='pca',
                     random_state=42).fit_transform(all_pts)
        feat_proj, proto_proj = tsne[:len(feats)], tsne[len(feats):]

        fig, ax = plt.subplots(1, 2 if _safe_umap is not None else 1,
                                figsize=(12, 5))
        axes = ax if hasattr(ax, '__len__') else [ax]
        scatter(axes[0], feat_proj, labs, proto_proj, title=f'Session {s} — t-SNE')
        umap_proj = _safe_umap(all_pts)
        if umap_proj is not None and len(axes) > 1:
            scatter(axes[1], umap_proj[:len(feats)], labs,
                     umap_proj[len(feats):], title=f'Session {s} — UMAP')
        fig.tight_layout()
        out = os.path.join(a.out_dir, f'session_{s}.png')
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f'saved {out}')

    # prototype drift
    if len(proto_history) >= 2:
        base_protos = proto_history[0]
        drifts = []
        for k in range(1, len(proto_history)):
            p = proto_history[k]
            m = min(base_protos.shape[0], p.shape[0])
            d = np.linalg.norm(p[:m] - base_protos[:m], axis=1)
            drifts.append(d.mean())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(a.session[1:], drifts, marker='o')
        ax.set_xlabel('session'); ax.set_ylabel('mean |Δ prototype|')
        ax.set_title('Base-class prototype drift')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, 'prototype_drift.png'), dpi=160)
        plt.close(fig)
        print('saved prototype_drift.png')


if __name__ == '__main__':
    main()
