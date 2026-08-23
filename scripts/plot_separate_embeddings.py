#!/usr/bin/env python3
"""
plot_separate_embeddings.py

Generate separate t-SNE (or UMAP fallback) plots for before and after checkpoints.

Usage:
  python baseline/scripts/plot_separate_embeddings.py \
    --ckpt_before baseline/save/base_train_for_meta.pth \
    --ckpt_after baseline/save/epoch_25.pth \
    --out_before baseline/before_tsne.png \
    --out_after baseline/after_tsne.png \
    --n_classes 10 --samples_per_class 150
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

def load_model(ckpt_path, args_model):
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

def extract_feats(model, loader, classes, samples_per_class):
    feats = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, lbl = batch[0], batch[1]
            else:
                continue
            if torch.cuda.is_available():
                data = data.cuda()
            if hasattr(model, 'encode'):
                out = model.encode(data)
            else:
                out = model(data)
                if isinstance(out, (list, tuple)):
                    out = out[0]
            if out.dim() > 2:
                out = out.mean(dim=[2,3])
            feats.append(out.cpu())
            labels.append(lbl.cpu())
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # sample per class
    sel_idx = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0: continue
        if len(idx) > samples_per_class:
            idx = np.random.choice(idx, samples_per_class, replace=False)
        sel_idx.extend(idx.tolist())
    sel_feats = feats[sel_idx]
    sel_labels = labels[sel_idx]
    centers = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            centers.append(np.zeros(feats.shape[1], dtype=float))
        else:
            centers.append(feats[idx].mean(axis=0))
    centers = np.vstack(centers)
    return sel_feats, sel_labels, centers

def embed_and_plot(X, labels, centers, classes, out_path, title=''):
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30, random_state=0)
    emb = tsne.fit_transform(X)
    n = labels.shape[0]
    emb_samples = emb[:n]
    emb_centers = emb[n:]

    plt.figure(figsize=(8,8))
    cmap = plt.get_cmap('tab20')
    for i, c in enumerate(classes):
        mask = labels == c
        pts = emb_samples[mask]
        if pts.size:
            plt.scatter(pts[:,0], pts[:,1], color=cmap(i%20), alpha=0.25, s=12)
    # plot centers
    for i, c in enumerate(classes):
        plt.scatter(emb_centers[i,0], emb_centers[i,1], marker='P', s=160, color=cmap(i%20), edgecolor='k')
        plt.text(emb_centers[i,0], emb_centers[i,1], str(int(c)), fontsize=12, fontweight='bold')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_before', default='baseline/save/base_train_for_meta.pth')
    parser.add_argument('--ckpt_after', default='baseline/save/epoch_25.pth')
    parser.add_argument('--out_before', default='baseline/before_tsne.png')
    parser.add_argument('--out_after', default='baseline/after_tsne.png')
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--samples_per_class', type=int, default=150)
    args = parser.parse_args()

    try:
        import train_unopenset as tu
    except Exception as e:
        print('Failed to import train_unopenset:', e)
        return

    parser_tu = tu.args_parser()
    args_ds = parser_tu.parse_known_args([])[0]
    if not hasattr(args_ds, 'dataloader'):
        args_ds.dataloader = SimpleNamespace()
        args_ds.dataloader.train_batch_size = 128
    # fill minimal defaults
    if not hasattr(args_ds, 'num_base'):
        args_ds.num_base = getattr(args_ds, 'num_labeled_classes', 80)
    if not hasattr(args_ds, 'num_labeled_classes'):
        args_ds.num_labeled_classes = 80
    tu.set_up_datasets(args_ds)
    full_dataset, full_loader = tu.get_pretrain_dataloader(args_ds)

    classes = np.unique(np.array([y for _, y in full_dataset]))[:args.n_classes]

    args_model = build_args_for_dataset()
    model_b = load_model(args.ckpt_before, args_model)
    model_a = load_model(args.ckpt_after, args_model)

    sb_feats, sb_labels, sb_centers = extract_feats(model_b, full_loader, classes, args.samples_per_class)
    sa_feats, sa_labels, sa_centers = extract_feats(model_a, full_loader, classes, args.samples_per_class)

    # embed separately (samples + centers appended)
    Xb = np.vstack([sb_feats, sb_centers])
    embed_and_plot(Xb, sb_labels, sb_centers, classes, args.out_before, title='Before')

    Xa = np.vstack([sa_feats, sa_centers])
    embed_and_plot(Xa, sa_labels, sa_centers, classes, args.out_after, title='After')

    print('Saved', args.out_before, 'and', args.out_after)

if __name__ == '__main__':
    main()
