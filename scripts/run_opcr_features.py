#!/usr/bin/env python
"""Paper-faithful OPCR transfer for frozen acoustic embeddings.

Implements reserved orthogonal targets, class-conditional Weibull tail rejection,
KMeans pseudo labels, and top-k confidence refinement. A closed-form linear map
replaces OPCR's image encoder updates and is explicitly recorded as a
paper-reimplementation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import weibull_min
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def means(x, y, labels):
    return np.stack([norm(x[y == c]).mean(0) for c in labels]).astype(np.float32)


def fit_map(source_anchors, targets, ridge):
    # Dual ridge solution: W = A^T(AA^T + lambda I)^-1 T.
    a = norm(source_anchors)
    gram = a @ a.T + ridge * np.eye(len(a), dtype=np.float32)
    return a.T @ np.linalg.solve(gram, targets[:len(a)])


def project(x, w):
    return norm(norm(x) @ w)


def nearest(z, targets):
    sim = z @ targets.T
    idx = sim.argmax(1)
    distance = np.sqrt(np.maximum(2 - 2 * sim[np.arange(len(z)), idx], 0))
    return idx, distance


def fit_evt(x_by_class, w, targets, tail_size):
    params = []
    for i, x in enumerate(x_by_class):
        d = np.linalg.norm(project(x, w) - targets[i], axis=1)
        tail = np.sort(d)[-min(tail_size, len(d)):]
        try:
            shape, _, scale = weibull_min.fit(tail, floc=0)
        except Exception:
            shape, scale = 2.0, max(float(tail.max()), 1e-3)
        params.append((float(shape), max(float(scale), 1e-4)))
    return params


def evt_scores(distance, index, params):
    return np.asarray([1.0 - weibull_min.cdf(d, params[i][0], loc=0, scale=params[i][1])
                       for d, i in zip(distance, index)])


def align(cluster, truth, novel):
    mat = np.zeros((len(novel), len(novel)), dtype=int)
    for i in range(len(novel)):
        for j, c in enumerate(novel):
            mat[i, j] = -np.sum((cluster == i) & (truth == c))
    r, c = linear_sum_assignment(mat)
    return {int(i): int(novel[j]) for i, j in zip(r, c)}


def accuracy(x, y, w, targets, labels):
    idx, _ = nearest(project(x, w), targets[:len(labels)])
    return float(np.mean(labels[idx] == y))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feature-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base', type=int, required=True)
    p.add_argument('--total', type=int, required=True)
    p.add_argument('--increment', type=int, default=5)
    p.add_argument('--sessions', type=int, default=5)
    p.add_argument('--ridge', type=float, default=0.05)
    p.add_argument('--evt-threshold', type=float, default=0.5)
    p.add_argument('--tail-size', type=int, default=20)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--seed', type=int, default=3420)
    a = p.parse_args()
    src, out = Path(a.feature_dir), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    s0 = np.load(src / 'session_0.npz')
    dim = s0['train_x'].shape[1]
    rng = np.random.default_rng(a.seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, a.total)))
    targets = q.T.astype(np.float32)
    labels = np.arange(a.base)
    source_anchors = means(s0['train_x'], s0['train_y'], labels)
    samples = [s0['train_x'][s0['train_y'] == c] for c in labels]
    w = fit_map(source_anchors, targets, a.ridge)
    evt = fit_evt(samples, w, targets, a.tail_size)
    records = [{'session': 0, 'all_acc': accuracy(s0['test_all_x'], s0['test_all_y'], w, targets, labels),
                'old_acc': accuracy(s0['test_all_x'], s0['test_all_y'], w, targets, labels),
                'novel_acc': 0.0, 'incremental_acc': 0.0}]

    for session in range(1, a.sessions):
        d = np.load(src / f'session_{session}.npz')
        x, y = d['train_x'], d['train_y'].astype(int)
        z = project(x, w)
        idx, distance = nearest(z, targets[:len(labels)])
        known_score = evt_scores(distance, idx, evt)
        pred_known = known_score >= a.evt_threshold
        true_known = y < a.base + (session - 1) * a.increment
        try:
            auroc = roc_auc_score(true_known, known_score)
        except ValueError:
            auroc = float('nan')
        f1 = f1_score(true_known, pred_known)
        candidate = np.flatnonzero(~pred_known)
        if len(candidate) < a.increment:
            candidate = np.argsort(known_score)[:a.increment]
        km = KMeans(a.increment, n_init=20, random_state=a.seed + session).fit(z[candidate])
        novel = np.arange(a.base + (session - 1) * a.increment, a.base + session * a.increment)
        mapping = align(km.labels_, y[candidate], novel)
        new_sources, new_samples = [], []
        for cls in novel:
            k = next(k for k, v in mapping.items() if v == cls)
            member_idx = candidate[km.labels_ == k]
            center = km.cluster_centers_[k]
            confidence = np.linalg.norm(z[member_idx] - center, axis=1)
            chosen = member_idx[np.argsort(confidence)[:min(a.topk, len(member_idx))]]
            new_sources.append(norm(x[chosen]).mean(0))
            new_samples.append(x[chosen])
        source_anchors = np.concatenate([source_anchors, np.asarray(new_sources)], axis=0)
        samples.extend(new_samples)
        labels = np.concatenate([labels, novel])
        w = fit_map(source_anchors, targets, a.ridge)
        evt = fit_evt(samples, w, targets, a.tail_size)
        inc_mask = d['test_all_y'] >= a.base
        rec = {'session': session, 'old_acc': accuracy(d['test_old_x'], d['test_old_y'], w, targets, labels),
               'novel_acc': accuracy(d['test_novel_x'], d['test_novel_y'], w, targets, labels),
               'incremental_acc': accuracy(d['test_all_x'][inc_mask], d['test_all_y'][inc_mask], w, targets, labels),
               'all_acc': accuracy(d['test_all_x'], d['test_all_y'], w, targets, labels),
               'auroc': float(auroc), 'osr_f1': float(f1), 'rejected': int(len(candidate))}
        records.append(rec); print(json.dumps(rec))

    result = {'method': 'OPCR-acoustic', 'provenance': 'paper-reimplementation',
              'config': vars(a), 'records': records}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
