#!/usr/bin/env python
"""YLOC-style prototype augmentation on common acoustic features.

The implementation follows YLOC's PAL principle: estimate each novel class as
an optimal-transport-weighted mixture of all old class distributions, sample
augmented features, confidence-weight them, and update the projection-free
cosine prototype bank. PCL is represented by base prototype centering because
the shared frozen checkpoint cannot be retrained without breaking feature
parity; this limitation is recorded in result provenance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def class_stats(x, y, labels):
    mus, vars_ = [], []
    for c in labels:
        z = norm(x[y == c])
        mus.append(z.mean(0)); vars_.append(z.var(0) + 1e-5)
    return np.asarray(mus, np.float32), np.asarray(vars_, np.float32)


def nearest(x, proto):
    z, p = norm(x), norm(proto)
    sim = z @ p.T
    idx = sim.argmax(1)
    distance = 1.0 - sim[np.arange(len(z)), idx]
    return idx, distance


def accuracy(x, y, labels, proto):
    idx, _ = nearest(x, proto)
    return float(np.mean(labels[idx] == y))


def align(cluster, truth, novel):
    cost = np.zeros((len(novel), len(novel)), dtype=int)
    for i in range(len(novel)):
        for j, c in enumerate(novel):
            cost[i, j] = -np.sum((cluster == i) & (truth == c))
    r, c = linear_sum_assignment(cost)
    return {int(i): int(novel[j]) for i, j in zip(r, c)}


def pal(raw, old_mu, old_var, temperature, raw_weight, n_aug, rng):
    raw = norm(raw)
    mu_n, var_n = raw.mean(0), raw.var(0) + 1e-5
    # Diagonal Gaussian 2-Wasserstein cost; entropy-regularized transport from
    # one novel distribution to all stored old distributions.
    w2 = ((old_mu - mu_n) ** 2 + old_var + var_n
          - 2.0 * np.sqrt(old_var * var_n)).sum(1)
    weight = softmax(-w2 / temperature)
    transfer_mu = weight @ old_mu
    transfer_var = weight @ old_var
    calibrated_mu = raw_weight * mu_n + (1.0 - raw_weight) * transfer_mu
    calibrated_var = raw_weight * var_n + (1.0 - raw_weight) * transfer_var
    synth = rng.normal(calibrated_mu, np.sqrt(calibrated_var), size=(n_aug, raw.shape[1]))
    synth = norm(synth)
    confidence = softmax((synth @ mu_n) / 0.1)
    aug_mu = confidence @ synth
    proto = norm((raw_weight * mu_n + (1.0 - raw_weight) * aug_mu)[None])[0]
    return proto, calibrated_var.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feature-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base', type=int, required=True)
    p.add_argument('--increment', type=int, default=5)
    p.add_argument('--sessions', type=int, default=5)
    p.add_argument('--radius-quantile', type=float, default=0.95)
    p.add_argument('--ot-temperature', type=float, default=0.5)
    p.add_argument('--raw-weight', type=float, default=0.8)
    p.add_argument('--n-aug', type=int, default=256)
    p.add_argument('--seed', type=int, default=3420)
    a = p.parse_args()
    src, out, rng = Path(a.feature_dir), Path(a.output), np.random.default_rng(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    s0 = np.load(src / 'session_0.npz')
    labels = np.arange(a.base)
    proto, vars_ = class_stats(s0['train_x'], s0['train_y'], labels)
    proto = norm(proto)
    radii = []
    for i, c in enumerate(labels):
        _, d = nearest(s0['train_x'][s0['train_y'] == c], proto[i:i + 1])
        radii.append(np.quantile(d, a.radius_quantile))
    radii = np.asarray(radii)
    records = [{'session': 0, 'all_acc': accuracy(s0['test_all_x'], s0['test_all_y'], labels, proto),
                'old_acc': accuracy(s0['test_all_x'], s0['test_all_y'], labels, proto),
                'novel_acc': 0.0, 'incremental_acc': 0.0}]

    for session in range(1, a.sessions):
        d = np.load(src / f'session_{session}.npz')
        x, y = d['train_x'], d['train_y'].astype(int)
        idx, distance = nearest(x, proto)
        score = radii[idx] - distance
        pred_known = score >= 0
        true_known = y < a.base + (session - 1) * a.increment
        try: auroc = roc_auc_score(true_known, score)
        except ValueError: auroc = float('nan')
        osr_f1 = f1_score(true_known, pred_known)
        candidate = np.flatnonzero(~pred_known)
        if len(candidate) < a.increment:
            candidate = np.argsort(score)[:a.increment]
        km = KMeans(a.increment, n_init=20, random_state=a.seed + session).fit(norm(x[candidate]))
        novel = np.arange(a.base + (session - 1) * a.increment, a.base + session * a.increment)
        mapping = align(km.labels_, y[candidate], novel)
        new_proto, new_var, new_radii = [], [], []
        for cls in novel:
            k = next(k for k, v in mapping.items() if v == cls)
            members = x[candidate][km.labels_ == k]
            p_new, v_new = pal(members, proto, vars_, a.ot_temperature,
                               a.raw_weight, a.n_aug, rng)
            new_proto.append(p_new); new_var.append(v_new)
            synth = norm(rng.normal(p_new, np.sqrt(v_new), size=(a.n_aug, len(p_new))))
            _, rd = nearest(synth, p_new[None])
            new_radii.append(np.quantile(rd, a.radius_quantile))
        proto = np.concatenate([proto, np.asarray(new_proto)], 0)
        vars_ = np.concatenate([vars_, np.asarray(new_var)], 0)
        radii = np.concatenate([radii, np.asarray(new_radii)])
        labels = np.concatenate([labels, novel])
        inc_mask = d['test_all_y'] >= a.base
        rec = {'session': session, 'old_acc': accuracy(d['test_old_x'], d['test_old_y'], labels, proto),
               'novel_acc': accuracy(d['test_novel_x'], d['test_novel_y'], labels, proto),
               'incremental_acc': accuracy(d['test_all_x'][inc_mask], d['test_all_y'][inc_mask], labels, proto),
               'all_acc': accuracy(d['test_all_x'], d['test_all_y'], labels, proto),
               'auroc': float(auroc), 'osr_f1': float(osr_f1), 'rejected': int(len(candidate))}
        records.append(rec); print(json.dumps(rec))

    result = {'method': 'YLOC-PAL-acoustic', 'provenance': 'paper-reimplementation',
              'limitations': 'PCL represented by frozen base prototype centering; PAL retained',
              'config': vars(a), 'records': records}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
