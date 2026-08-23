#!/usr/bin/env python3
"""Train a base/novel router on held-out FSC classes 59--68 only."""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def score_features(query, base_proto, novel_proto, novel_support=None):
    q = F.normalize(query, dim=1)
    b = q @ F.normalize(base_proto, dim=1).t()
    n = q @ F.normalize(novel_proto, dim=1).t()
    bt = b.topk(2, dim=1).values
    nt = n.topk(2, dim=1).values
    values = [bt[:, 0], nt[:, 0], bt[:, 0] - bt[:, 1],
              nt[:, 0] - nt[:, 1], nt[:, 0] - bt[:, 0]]
    if novel_support is not None:
        support_score = q @ F.normalize(novel_support, dim=1).t()
        support_top = support_score.topk(min(2, support_score.size(1)), dim=1).values
        values.extend([support_top[:, 0], support_top.mean(dim=1)])
    return torch.stack(values, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=3420)
    p.add_argument('--episodes', type=int, default=100)
    a = p.parse_args()
    g = torch.load(a.geometry, map_location='cpu', weights_only=True)
    c = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    state = c.get('params', c.get('cls_params', c))
    pools = [x.float() for x in g['class_features']]
    rng = np.random.default_rng(a.seed)
    train_x, train_y = [], []
    for _ in range(a.episodes):
        pseudo_novel = sorted(rng.choice(59, size=10, replace=False).tolist())
        pseudo_base = [i for i in range(59) if i not in pseudo_novel]
        episode_base = state['fc.weight'][pseudo_base].float()
        episode_novel = torch.stack([pools[i][:5].mean(0) for i in pseudo_novel])
        bq = torch.cat([pools[i][5:10] for i in pseudo_base])
        nq = torch.cat([pools[i][5:25] for i in pseudo_novel])
        train_x.extend([score_features(bq, episode_base, episode_novel),
                        score_features(nq, episode_base, episode_novel)])
        train_y.extend([torch.zeros(len(bq)), torch.ones(len(nq))])
    fit_x = torch.cat(train_x).numpy()
    fit_y = torch.cat(train_y).numpy().astype(np.int64)
    clf = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000,
                             random_state=a.seed).fit(fit_x, fit_y)

    # One-based 60--69 == zero-based 59--68, reserved for model selection only.
    heldout = list(range(59, 69))
    base_proto = state['fc.weight'][:59].float()
    novel_proto = torch.stack([pools[i][:5].mean(0) for i in heldout])
    base_query = torch.cat([pools[i] for i in range(59)])
    novel_query = torch.cat([pools[i][5:] for i in heldout])
    x = torch.cat([score_features(base_query, base_proto, novel_proto),
                   score_features(novel_query, base_proto, novel_proto)]).numpy()
    y = np.r_[np.zeros(len(base_query), dtype=np.int64),
              np.ones(len(novel_query), dtype=np.int64)]
    pred = clf.predict(x)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    torch.save({'coef': torch.from_numpy(clf.coef_[0]).float(),
                'intercept': float(clf.intercept_[0]),
                'feature_order': ['base_top1', 'novel_top1', 'base_gap', 'novel_gap', 'novel_minus_base'],
                'validation_classes_zero_based': heldout, 'offline': True,
                'encoder': 'current model.encode'}, a.output)
    print(f'train_samples={len(fit_y)} val_samples={len(y)} '
          f'balanced_acc={balanced_accuracy_score(y, pred):.6f} '
          f'coef={clf.coef_[0].tolist()} intercept={clf.intercept_[0]:.6f} output={a.output}')


if __name__ == '__main__':
    main()
