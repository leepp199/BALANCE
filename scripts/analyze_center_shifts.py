#!/usr/bin/env python3
"""
analyze_center_shifts.py

Compute per-class center L2 shifts between two checkpoints and save CSV.
Also produce a UMAP (or t-SNE fallback) joint embedding plot of samples+centers.

Usage:
  python baseline/scripts/analyze_center_shifts.py \
    --ckpt_before baseline/save/base_train_for_meta.pth \
    --ckpt_after baseline/save/epoch_25.pth \
    --out_csv baseline/center_shifts.csv \
    --out_plot baseline/center_shifts_umap.png
"""
import os
import sys
from types import SimpleNamespace
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)


def build_args_for_dataset():
    a = SimpleNamespace()
    a.dataloader = SimpleNamespace()
    a.dataloader.train_batch_size = 128
    a.extractor = SimpleNamespace()
    a.extractor.window_size = 400
    a.extractor.hop_size = 160
    a.extractor.window = 'hann'
    a.extractor.sample_rate = 16000
    a.extractor.mel_bins = 128
    a.extractor.fmin = 0
    a.extractor.fmax = 8000
    a.n_ways = 10
    a.train_weight_base = 0.0
    a.base_seman_calib = True
    a.neg_gen_type = 'att'
    a.agg = 'avg'
    a.network = SimpleNamespace()
    a.network.new_mode = 'cos'
    a.network.temperature = 10.0
    a.stdu = SimpleNamespace()
    a.stdu.num_tmpb = 80
    return a


def load_encoder_model(ckpt_path, args_model):
    import importlib
    network = importlib.import_module('network')
    Model = getattr(network, 'MYNET')
    a = SimpleNamespace()
    a.n_ways = getattr(args_model, 'n_ways', 10)
    a.base_seman_calib = getattr(args_model, 'base_seman_calib', True)
    a.neg_gen_type = getattr(args_model, 'neg_gen_type', 'att')
    a.agg = getattr(args_model, 'agg', 'avg')
    a.train_weight_base = getattr(args_model, 'train_weight_base', 0.0)
    a.extractor = args_model.extractor
    a.network = getattr(args_model, 'network', SimpleNamespace(new_mode='cos', temperature=10.0))
    a.stdu = getattr(args_model, 'stdu', SimpleNamespace(num_tmpb=80))
    model = Model(a, mode='encoder')
    if torch.cuda.is_available():
        model = model.cuda()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = ck.get('params', ck.get('state_dict', ck))
    if isinstance(sd, dict):
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            model_sd = model.state_dict()
            filtered = {k: v for k, v in sd.items() if k in model_sd}
            model_sd.update(filtered)
            model.load_state_dict(model_sd, strict=False)
    model.eval()
    return model


def extract_features(model, loader):
    feats = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, lbl = batch[0], batch[1]
            else:
                continue
            if torch.cuda.is_available():
                data = data.cuda()
            # encoder exposes encode in MYNET
            if hasattr(model, 'encode'):
                out = model.encode(data)
            else:
                out = model(data)
                if isinstance(out, (list, tuple)):
                    out = out[0]
            if out.dim() > 2:
                out = out.mean(dim=[2, 3])
            feats.append(out.cpu())
            labels.append(lbl.cpu())
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return feats, labels


def compute_centers(feats, labels, classes):
    centers = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            centers.append(np.zeros(feats.shape[1], dtype=float))
        else:
            centers.append(feats[idx].mean(axis=0))
    return np.vstack(centers)


def try_umap(X, n_neighbors=15, min_dist=0.1):
    try:
        import umap.umap_ as umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
        emb = reducer.fit_transform(X)
        return emb, 'umap'
    except Exception:
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30, random_state=0)
        emb = tsne.fit_transform(X)
        return emb, 'tsne'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_before', default='baseline/save/base_train_for_meta.pth')
    parser.add_argument('--ckpt_after', default='baseline/save/epoch_25.pth')
    parser.add_argument('--out_csv', default='baseline/center_shifts.csv')
    parser.add_argument('--out_plot', default='baseline/center_shifts_umap.png')
    parser.add_argument('--n_classes', type=int, default=20)
    parser.add_argument('--samples_per_class', type=int, default=80, help='samples per class to plot')
    args = parser.parse_args()

    try:
        import train_unopenset as tu
    except Exception as e:
        print('Failed to import train_unopenset:', e)
        return

    parser_tu = tu.args_parser()
    args_ds = parser_tu.parse_known_args([])[0]
    # fill commonly required defaults
    if not hasattr(args_ds, 'dataloader'):
        args_ds.dataloader = SimpleNamespace()
        args_ds.dataloader.train_batch_size = 128
    if not hasattr(args_ds, 'num_base'):
        args_ds.num_base = getattr(args_ds, 'num_labeled_classes', 80)
    if not hasattr(args_ds, 'num_labeled_classes'):
        args_ds.num_labeled_classes = 80
    if not hasattr(args_ds, 'num_unlabeled_classes'):
        args_ds.num_unlabeled_classes = 5
    if not hasattr(args_ds, 'n_ways'):
        args_ds.n_ways = getattr(args_ds, 'n_ways', 10)
    if not hasattr(args_ds, 'n_shots'):
        args_ds.n_shots = getattr(args_ds, 'n_shots', 5)
    if not hasattr(args_ds, 'n_queries'):
        args_ds.n_queries = getattr(args_ds, 'n_queries', 15)
    tu.set_up_datasets(args_ds)
    full_dataset, full_loader = tu.get_pretrain_dataloader(args_ds)

    args_model = build_args_for_dataset()
    model_b = load_encoder_model(args.ckpt_before, args_model)
    model_a = load_encoder_model(args.ckpt_after, args_model)

    print('Extracting features (before)...')
    feats_b, labels = extract_features(model_b, full_loader)
    print('Extracting features (after)...')
    feats_a, labels_a = extract_features(model_a, full_loader)
    assert labels.shape == labels_a.shape

    classes = np.unique(labels)
    if len(classes) > args.n_classes:
        classes = classes[np.linspace(0, len(classes)-1, args.n_classes, dtype=int)]

    centers_b = compute_centers(feats_b, labels, classes)
    centers_a = compute_centers(feats_a, labels_a, classes)

    shifts = np.linalg.norm(centers_a - centers_b, axis=1)
    out_rows = []
    for i, c in enumerate(classes):
        out_rows.append((int(c), float(np.linalg.norm(centers_b[i])), float(np.linalg.norm(centers_a[i])), float(shifts[i])))

    import csv
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'norm_before', 'norm_after', 'l2_shift'])
        for r in out_rows:
            w.writerow(r)

    print('Saved CSV to', args.out_csv)

    # joint embedding: sample subset to keep plot readable
    samp_per = args.samples_per_class
    sel_idx = []
    sel_labels = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) > samp_per:
            idx = np.random.choice(idx, samp_per, replace=False)
        sel_idx.extend(idx.tolist())
        sel_labels.extend([int(c)] * len(idx))

    sel_feats_b = feats_b[sel_idx]
    sel_feats_a = feats_a[sel_idx]

    X = np.vstack([sel_feats_b, sel_feats_a, centers_b, centers_a])
    emb, method = try_umap(X)
    n_b = sel_feats_b.shape[0]
    n_a = sel_feats_a.shape[0]
    n_c = centers_b.shape[0]
    b_emb = emb[:n_b]
    a_emb = emb[n_b:n_b + n_a]
    bc_emb = emb[n_b + n_a:n_b + n_a + n_c]
    ac_emb = emb[n_b + n_a + n_c:]

    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('tab20')
    # plot before samples faint
    for i, c in enumerate(classes):
        mask = np.array(sel_labels) == int(c)
        pts = b_emb[mask]
        if pts.size:
            plt.scatter(pts[:, 0], pts[:, 1], color=cmap(i % 20), alpha=0.15, s=10)
    # after samples
    for i, c in enumerate(classes):
        mask = np.array(sel_labels) == int(c)
        pts = a_emb[mask]
        if pts.size:
            plt.scatter(pts[:, 0], pts[:, 1], color=cmap(i % 20), alpha=0.25, s=10)
    # centers
    for i, c in enumerate(classes):
        plt.scatter(bc_emb[i, 0], bc_emb[i, 1], marker='P', s=140, color=cmap(i % 20), edgecolor='k')
        plt.scatter(ac_emb[i, 0], ac_emb[i, 1], marker='X', s=140, color=cmap(i % 20), edgecolor='k')
        dx, dy = ac_emb[i] - bc_emb[i]
        plt.arrow(bc_emb[i, 0], bc_emb[i, 1], dx, dy, color='red', width=0.001, head_width=4.0, head_length=4.0, length_includes_head=True)
        plt.text(bc_emb[i, 0], bc_emb[i, 1], str(int(c)), fontsize=12, fontweight='bold')

    plt.title(f'Joint embedding and center shifts ({method})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    print('Saved plot to', args.out_plot)


if __name__ == '__main__':
    main()
