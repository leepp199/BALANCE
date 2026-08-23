#!/usr/bin/env python
"""Happy hardness-aware prototype-augmentation transfer on frozen audio features.

The official NeurIPS-2024 implementation trains an image encoder with feature
distillation, group-wise entropy regularization, and hardness-aware prototype
augmentation.  Under the common frozen-audio-feature protocol, this transfer keeps
the official prototype/radius/hardness sampling equations and optimizes only the
cosine classifier.  It is therefore reported as a component reimplementation, not
as an official-code reproduction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def means(x, y, labels):
    return np.stack([norm(x[y == c]).mean(0) for c in labels]).astype(np.float32)


def nearest(x, weight):
    sim = norm(x) @ norm(weight).T
    idx = sim.argmax(1)
    return idx, 1.0 - sim[np.arange(len(sim)), idx]


def accuracy(x, y, labels, weight):
    idx, _ = nearest(x, weight)
    return float(np.mean(labels[idx] == y))


def align(cluster, truth, novel):
    cost = np.zeros((len(novel), len(novel)), dtype=int)
    for i in range(len(novel)):
        for j, c in enumerate(novel):
            cost[i, j] = -np.sum((cluster == i) & (truth == c))
    row, col = linear_sum_assignment(cost)
    return {int(i): int(novel[j]) for i, j in zip(row, col)}


def happy_update(weight, radius, steps, batch, temperature, lr, seed, device):
    """Official hardness sampling plus a projection-free cosine classifier update."""
    torch.manual_seed(seed)
    p = F.normalize(torch.as_tensor(weight, dtype=torch.float32, device=device), dim=-1)
    similarity = p @ p.T
    similarity.fill_diagonal_(0.0)
    mean_similarity = similarity.sum(1) / max(len(p) - 1, 1)
    probability = F.softmax(mean_similarity / temperature, dim=0)
    classifier = p.detach().clone().requires_grad_(True)
    anchor = classifier.detach().clone()
    optimizer = torch.optim.SGD([classifier], lr=lr, momentum=0.9)
    for _ in range(steps):
        target = torch.multinomial(probability, batch, replacement=True)
        synthetic = p[target] + torch.randn(batch, p.shape[1], device=device) * radius
        logits = F.normalize(synthetic, dim=-1) @ F.normalize(classifier, dim=-1).T / 0.1
        # Frozen-feature analogue of Happy's feature-distillation protection.
        loss = F.cross_entropy(logits, target) + 2.0 * F.mse_loss(classifier, anchor)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return F.normalize(classifier.detach(), dim=-1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feature-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base', type=int, required=True)
    p.add_argument('--increment', type=int, default=5)
    p.add_argument('--sessions', type=int, default=5)
    p.add_argument('--radius-quantile', type=float, default=0.95)
    p.add_argument('--radius-scale', type=float, default=1.0)
    p.add_argument('--hardness-temperature', type=float, default=0.1)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=3420)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    src, out = Path(a.feature_dir), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    s0 = np.load(src / 'session_0.npz')
    labels = np.arange(a.base)
    weight = norm(means(s0['train_x'], s0['train_y'], labels))
    per_class_variance = [norm(s0['train_x'][s0['train_y'] == c]).var(0).mean()
                          for c in labels]
    radius = float(np.sqrt(np.mean(per_class_variance)) * a.radius_scale)
    radii = []
    for i, c in enumerate(labels):
        _, distance = nearest(s0['train_x'][s0['train_y'] == c], weight[i:i + 1])
        radii.append(np.quantile(distance, a.radius_quantile))
    radii = np.asarray(radii)
    records = [{'session': 0,
                'all_acc': accuracy(s0['test_all_x'], s0['test_all_y'], labels, weight),
                'old_acc': accuracy(s0['test_all_x'], s0['test_all_y'], labels, weight),
                'novel_acc': 0.0, 'incremental_acc': 0.0}]

    for session in range(1, a.sessions):
        d = np.load(src / f'session_{session}.npz')
        x, y = d['train_x'], d['train_y'].astype(int)
        index, distance = nearest(x, weight)
        score = radii[index] - distance
        predicted_known = score >= 0
        true_known = y < a.base + (session - 1) * a.increment
        try:
            auroc = roc_auc_score(true_known, score)
        except ValueError:
            auroc = float('nan')
        osr_f1 = f1_score(true_known, predicted_known)
        candidate = np.flatnonzero(~predicted_known)
        if len(candidate) < a.increment:
            candidate = np.argsort(score)[:a.increment]
        km = KMeans(a.increment, n_init=20, random_state=a.seed + session).fit(norm(x[candidate]))
        novel = np.arange(a.base + (session - 1) * a.increment,
                          a.base + session * a.increment)
        mapping = align(km.labels_, y[candidate], novel)
        new_weight, new_radii = [], []
        for cls in novel:
            cluster = next(k for k, value in mapping.items() if value == cls)
            member = x[candidate][km.labels_ == cluster]
            prototype = norm(member).mean(0)
            new_weight.append(prototype)
            _, rd = nearest(member, prototype[None])
            new_radii.append(np.quantile(rd, a.radius_quantile))
        weight = np.concatenate([weight, norm(np.asarray(new_weight))], 0)
        radii = np.concatenate([radii, np.asarray(new_radii)])
        labels = np.concatenate([labels, novel])
        weight = happy_update(weight, radius, a.steps, a.batch_size,
                              a.hardness_temperature, a.lr,
                              a.seed + session, torch.device(a.device))
        incremental = d['test_all_y'] >= a.base
        record = {
            'session': session,
            'old_acc': accuracy(d['test_old_x'], d['test_old_y'], labels, weight),
            'novel_acc': accuracy(d['test_novel_x'], d['test_novel_y'], labels, weight),
            'incremental_acc': accuracy(d['test_all_x'][incremental], d['test_all_y'][incremental], labels, weight),
            'all_acc': accuracy(d['test_all_x'], d['test_all_y'], labels, weight),
            'auroc': float(auroc), 'osr_f1': float(osr_f1), 'rejected': int(len(candidate))}
        records.append(record); print(json.dumps(record))

    result = {
        'method': 'Happy-HAProto-acoustic',
        'provenance': 'official-component-reimplementation',
        'limitations': ('Official hardness-aware prototype/radius sampling retained; '
                        'image encoder, multi-view entropy training and feature distillation '
                        'replaced by a frozen audio encoder and anchored cosine-head update.'),
        'official_source': 'https://github.com/mashijie1028/Happy-CGCD',
        'config': vars(a), 'records': records}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
