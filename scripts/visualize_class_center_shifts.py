"""
visualize_class_center_shifts.py

Generate a twin-panel t-SNE visualization showing per-class sample clusters
and arrows from each class center (before -> after).

Usage:
  python baseline/scripts/visualize_class_center_shifts.py \
    --ckpt_before /data/lqq/baseline/save/base_train_for_meta.pth \
    --ckpt_after  /data/lqq/baseline/save/epoch_25.pth \
    --out /data/lqq/baseline/class_center_shifts.png

The script extracts encoder features from the dataset using functions in
`train_unopenset.py` and `network.MYNET` (mode='encoder'). It samples up to
`--samples_per_class` points per class, computes class centers before/after,
and plots them with arrows showing shifts.
"""

import os
import sys
import argparse
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)


def build_args_for_dataset():
    # Minimal args object expected by train_unopenset functions
    args = SimpleNamespace()
    args.dataset = 'librispeech'
    args.dataroot = '/data/datasets/librispeech_fscil/'
    args.dataloader = SimpleNamespace()
    args.dataloader.train_batch_size = 128
    # used by MYNET constructor
    args.train_weight_base = 0.0
    # few optional defaults
    args.n_ways = 10
    args.n_shots = 5
    args.n_queries = 15
    return args


def load_encoder_model(ckpt_path, args_model):
    # Instantiate full MYNET in 'encoder' mode but provide minimal args so Classifier init succeeds
    import importlib
    network = importlib.import_module('network')
    Model = getattr(network, 'MYNET')

    # build a minimal args object with commonly accessed fields
    a = SimpleNamespace()
    a.n_ways = getattr(args_model, 'n_ways', 10)
    a.base_seman_calib = getattr(args_model, 'base_seman_calib', True)
    a.neg_gen_type = getattr(args_model, 'neg_gen_type', 'att')
    a.agg = getattr(args_model, 'agg', 'avg')
    a.train_weight_base = getattr(args_model, 'train_weight_base', 0.0)
    # nested extractor params required by set_module_for_audio
    ext = SimpleNamespace()
    ext.window_size = getattr(args_model, 'window_size', 400)
    ext.hop_size = getattr(args_model, 'hop_size', 160)
    ext.window = getattr(args_model, 'window', 'hann')
    ext.sample_rate = getattr(args_model, 'sample_rate', 16000)
    ext.mel_bins = getattr(args_model, 'mel_bins', 128)
    ext.fmin = getattr(args_model, 'fmin', 0)
    ext.fmax = getattr(args_model, 'fmax', 8000)
    a.extractor = ext

    # some code paths expect nested attrs like args.network and args.stdu; provide minimal placeholders
    a.network = SimpleNamespace()
    a.network.new_mode = 'cos'
    a.network.temperature = 10.0
    a.stdu = SimpleNamespace()
    a.stdu.num_tmpb = getattr(a, 'num_base', 80)

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
            # try filtering keys that exist in model
            model_sd = model.state_dict()
            filtered = {k: v for k, v in sd.items() if k in model_sd}
            model_sd.update(filtered)
            model.load_state_dict(model_sd, strict=False)
    model.eval()
    return model


def extract_features_for_checkpoint(model, full_loader):
    feats = []
    labels = []
    with torch.no_grad():
        for batch in full_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, lbl = batch[0], batch[1]
            else:
                # unsupported batch format
                continue
            if torch.cuda.is_available():
                data = data.cuda()
            # support encoder implementations that expose `encode` or use forward
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


def sample_per_class(feats, labels, classes, samples_per_class):
    sel_feats = []
    sel_labels = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) > samples_per_class:
            idx = np.random.choice(idx, samples_per_class, replace=False)
        sel_feats.append(feats[idx])
        sel_labels.extend([c] * len(idx))
    if len(sel_feats) == 0:
        return np.zeros((0, feats.shape[1])), np.array([])
    sel_feats = np.vstack(sel_feats)
    sel_labels = np.array(sel_labels)
    return sel_feats, sel_labels


def plot_shift(before_feats, after_feats, before_labels, classes, before_centers, after_centers, out,
               perplexity=30, pca_init=True, alpha=0.35, arrow_scale=1.0, figsize=(14, 8)):
    # Embed samples and centers jointly so centers have accurate embedded coords.
    # Order: [before_samples, after_samples, before_centers, after_centers]
    X = np.vstack([before_feats, after_feats, before_centers, after_centers])
    n_before = before_feats.shape[0]
    n_after = after_feats.shape[0]
    n_cent = before_centers.shape[0]

    tsne = TSNE(n_components=2, init='pca' if pca_init else 'random', learning_rate='auto',
                perplexity=perplexity, random_state=0)
    emb = tsne.fit_transform(X)

    b_emb = emb[:n_before]
    a_emb = emb[n_before:n_before + n_after]
    bc_emb = emb[n_before + n_after:n_before + n_after + n_cent]
    ac_emb = emb[n_before + n_after + n_cent:]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    cmap = plt.get_cmap('tab20')

    # plot samples
    for ax, emb_pts, title in zip(axes, (b_emb, a_emb), ('Before', 'After')):
        for i, c in enumerate(classes):
            mask = (before_labels == c)
            if emb_pts is a_emb:
                # for after panel, labels are same ordering as before_labels used for sampling
                mask = (before_labels == c)
            pts = emb_pts[mask]
            if pts.size == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], color=cmap(i % 20), alpha=alpha, s=20, edgecolors='none')
        ax.set_title(title)
        ax.axis('off')

    # plot centers and arrows
    for i, c in enumerate(classes):
        bc_pos = bc_emb[i]
        ac_pos = ac_emb[i]
        color = cmap(i % 20)
        # draw centers
        axes[0].scatter(bc_pos[0], bc_pos[1], marker='P', s=220, color=color, edgecolor='k', zorder=5)
        axes[1].scatter(ac_pos[0], ac_pos[1], marker='P', s=220, color=color, edgecolor='k', zorder=5)
        # draw arrow on combined visual (left panel shows arrow from before->after)
        dx, dy = (ac_pos - bc_pos) * arrow_scale
        axes[0].arrow(bc_pos[0], bc_pos[1], dx, dy, color='red', linewidth=1.2,
                      head_width=6.0 * arrow_scale, head_length=6.0 * arrow_scale, length_includes_head=True, zorder=4)
        # label near before center
        axes[0].text(bc_pos[0], bc_pos[1], str(c), fontsize=14, fontweight='bold', va='bottom')

    plt.suptitle('Class center shifts (before -> after)')
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_before', type=str, required=False,
                        default='/data/lqq/baseline/save/base_train_for_meta.pth')
    parser.add_argument('--ckpt_after', type=str, required=False,
                        default='/data/lqq/baseline/save/epoch_25.pth')
    parser.add_argument('--out', type=str, default='/data/lqq/baseline/class_center_shifts.png')
    parser.add_argument('--n_classes', type=int, default=12)
    parser.add_argument('--samples_per_class', type=int, default=60)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--pca_init', action='store_true', help='use PCA init for t-SNE')
    parser.add_argument('--alpha', type=float, default=0.35, help='marker alpha')
    parser.add_argument('--arrow_scale', type=float, default=1.0, help='scale for arrow lengths')
    parser.add_argument('--fig_w', type=float, default=14.0, help='figure width')
    parser.add_argument('--fig_h', type=float, default=8.0, help='figure height')
    args = parser.parse_args()

    # import train_unopenset helpers and build args via its parser (has many defaults)
    try:
        import train_unopenset as tu
    except Exception as e:
        print('Failed to import train_unopenset:', e)
        return

    try:
        parser = tu.args_parser()
        args_ds = parser.parse_known_args([])[0]
        # ensure dataloader settings exist
        if not hasattr(args_ds, 'dataloader'):
            args_ds.dataloader = SimpleNamespace()
            args_ds.dataloader.train_batch_size = 128
        # provide fallback defaults for commonly used attrs
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
    except Exception as e:
        print('Failed to build dataloader via train_unopenset:', e)
        return

    # load models
    args_model = build_args_for_dataset()
    model_before = load_encoder_model(args.ckpt_before, args_model)
    model_after = load_encoder_model(args.ckpt_after, args_model)

    print('Extracting features (before)...')
    feats_before, labels = extract_features_for_checkpoint(model_before, full_loader)
    print('Extracting features (after)...')
    feats_after, labels_after = extract_features_for_checkpoint(model_after, full_loader)

    assert labels.shape == labels_after.shape

    classes = np.unique(labels)
    if len(classes) > args.n_classes:
        # pick evenly spaced classes
        classes = classes[np.linspace(0, len(classes)-1, args.n_classes, dtype=int)]

    sb_before, sl_before = sample_per_class(feats_before, labels, classes, args.samples_per_class)
    sb_after, sl_after = sample_per_class(feats_after, labels_after, classes, args.samples_per_class)

    # compute centers in feature space (before/after)
    centers_before = []
    centers_after = []
    for c in classes:
        centers_before.append(feats_before[labels == c].mean(axis=0))
        centers_after.append(feats_after[labels_after == c].mean(axis=0))
    centers_before = np.vstack(centers_before)
    centers_after = np.vstack(centers_after)

    # Plot
    plot_shift(sb_before, sb_after, sl_before, classes, centers_before, centers_after, args.out,
               perplexity=args.perplexity, pca_init=args.pca_init, alpha=args.alpha,
               arrow_scale=args.arrow_scale, figsize=(args.fig_w, args.fig_h))
    print('Saved', args.out)


if __name__ == '__main__':
    main()
