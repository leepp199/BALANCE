"""Task 3.2 — OSR score histograms, ROC and confusion matrix."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from network import MYNET, replace_base_fc  # noqa: E402
from data.dataloader import get_dataloader, get_pretrain_dataloader  # noqa: E402
from models.baselines import build_osr  # noqa: E402
from scripts.run_all_baselines import encode_loader, load_args  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yml')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--osr', nargs='+', default=['mls', 'tane', 'nci'])
    p.add_argument('--session', type=int, default=1)
    p.add_argument('--out_dir', default='save/figures/osr')
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

    loader = get_dataloader(args, a.session)[1]
    feats, labs = encode_loader(model, loader, dev)
    feats = feats.to(dev)
    gt_unknown = (labs >= args.num_base).numpy().astype(int)

    protos = model.fc.weight[:args.num_base, :].detach().to(dev)

    fig_h, axh = plt.subplots(1, len(a.osr), figsize=(4.2 * len(a.osr), 3.5))
    fig_r, axr = plt.subplots(figsize=(5, 5))
    axh = [axh] if len(a.osr) == 1 else axh

    for i, name in enumerate(a.osr):
        osr = build_osr(name, args)
        s = osr.score(feats, protos).cpu().numpy()
        # histogram
        axh[i].hist(s[gt_unknown == 0], bins=40, alpha=0.6, label='known')
        axh[i].hist(s[gt_unknown == 1], bins=40, alpha=0.6, label='unknown')
        axh[i].set_title(f'{name.upper()}'); axh[i].legend()
        # ROC
        fpr, tpr, _ = roc_curve(gt_unknown, s)
        axr.plot(fpr, tpr, label=f'{name.upper()} AUC={auc(fpr, tpr):.3f}')

    axr.plot([0, 1], [0, 1], ls='--', color='gray')
    axr.set_xlabel('FPR'); axr.set_ylabel('TPR')
    axr.set_title(f'Session {a.session} — OSR ROC'); axr.legend()

    fig_h.tight_layout(); fig_r.tight_layout()
    fig_h.savefig(os.path.join(a.out_dir, f'score_hist_s{a.session}.png'), dpi=160)
    fig_r.savefig(os.path.join(a.out_dir, f'roc_s{a.session}.png'), dpi=160)
    plt.close('all')
    print('saved OSR histograms and ROC.')

    # confusion matrix on closed-set (known labels only)
    kn_mask = gt_unknown == 0
    if kn_mask.any():
        logits = F.cosine_similarity(feats[kn_mask].unsqueeze(1), protos, dim=-1)
        preds = logits.argmax(dim=1).cpu().numpy()
        cm = confusion_matrix(labs.numpy()[kn_mask], preds,
                                labels=np.arange(args.num_base))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xlabel('pred'); ax.set_ylabel('true')
        ax.set_title(f'Session {a.session} — known-class confusion')
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, f'confusion_s{a.session}.png'), dpi=160)
        plt.close(fig)
        print('saved confusion matrix.')


if __name__ == '__main__':
    main()
