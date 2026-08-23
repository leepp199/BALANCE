"""Training and evaluation pipeline for baseline comparison.

Runs 5-session CIL + OSR evaluation on a given dataset.
Each CIL method owns its own encoder (standard ResNet18, NOT MYNET).

Usage:
    python -m repro_baselines.train --cil prototypical --osr mls \\
        --dataset librispeech --dataroot /data/datasets/librispeech_fscil/ \\
        --num_base 80 --num_novel 20 --num_all 100
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Fix soundfile multiprocessing fork issue
try:
    torchaudio.set_audio_backend('soundfile')
except Exception:
    pass
os.environ['LIBSNDFILE_PREVENT_FORK'] = '1'

# Add project root
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from repro_baselines.methods.cil import build_cil, CIL_REGISTRY
from repro_baselines.methods.osr import build_osr, OSR_REGISTRY


# ======================================================================
# Data utilities
# ======================================================================

def build_base_loader(dataset, dataroot, num_base, args):
    """Build dataloader for base training (session 0)."""
    from data.dataloader import get_base_dataloader_stdu
    from addict import Dict as aDict

    cfg = aDict({
        'dataset': dataset,
        'dataroot': dataroot,
        'num_base': num_base,
        'tmp_train': True,
        'stdu': {
            'num_tmpb': min(55, num_base),
            'num_tmpi': 25,
            'num_tmps': 14,
            'num_incre': 5,
        },
        'episode': {
            'train_episode': 50,
            'low_way': max(5, min(20, num_base // 4)),
            'low_shot': 5,
            'episode_way': 5,
            'episode_shot': 5,
            'episode_query': 15,
        },
        'dataloader': {
            'num_workers': 4,
            'train_batch_size': 128,
        },
    })
    for k, v in vars(args).items():
        cfg[k] = v
    cfg.Dataset = args.Dataset

    return get_base_dataloader_stdu(cfg)


def build_inc_loader(dataset, dataroot, num_base, session, args, way=5):
    """Build dataloader for incremental session."""
    from data.dataloader import get_new_dataloader
    from addict import Dict as aDict

    cfg = aDict({
        'dataset': dataset,
        'dataroot': dataroot,
        'num_base': num_base,
        'way': way,
        'seq_sample': False,
        'episode': {
            'episode_way': way,
            'episode_shot': 5,
            'episode_query': 15,
        },
        'dataloader': {
            'num_workers': 4,
        },
    })
    for k, v in vars(args).items():
        cfg[k] = v
    cfg.Dataset = args.Dataset

    return get_new_dataloader(cfg, session)


def build_test_loader(dataset, dataroot, num_base, session, args):
    """Build test loader for all encountered classes up to session."""
    from data.dataloader import get_testloader
    from addict import Dict as aDict

    cfg = aDict({
        'dataset': dataset,
        'dataroot': dataroot,
        'num_base': num_base,
        'way': 5,
        'dataloader': {
            'num_workers': 4,
            'test_batch_size': 100,
        },
    })
    for k, v in vars(args).items():
        cfg[k] = v
    cfg.Dataset = args.Dataset

    return get_testloader(cfg, session)


def init_audio_dataset(dataset):
    """Import the right Dataset module."""
    if 'nsynth' in dataset.lower():
        from data.nsynth import NDS, Opennds as OpenDS
    elif 'fmc' in dataset.lower() or 'fsc' in dataset.lower():
        from data.FMC import FSDCLIPS, Openfs as OpenDS
    else:
        from data.librispeech import LBRS, Openlbrs as OpenDS
    return LBRS, OpenDS


# ======================================================================
# Feature extraction — now using cil_method's own model
# ======================================================================

def extract_feats(model, x):
    """Extract features using cil_method.model.encode()."""
    return model.encode(x)


@torch.no_grad()
def compute_acc(model, cil_method, test_loader, n_known, device):
    """Compute classification accuracy."""
    model.eval()
    cil_method.eval()
    correct, total = 0, 0
    for batch in test_loader:
        x, y = batch
        if isinstance(x, (list, tuple)):
            x = x[0]
        x, y = x.to(device), y.to(device)
        feats = extract_feats(model, x)
        logits = cil_method.classify(feats, n_known)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def compute_osr_metrics(model, cil_method, osr_method,
                        test_loader, n_known, device):
    """Compute AUROC and FPR95 for open-set detection."""
    model.eval()
    cil_method.eval()

    all_scores = []
    all_labels = []
    for batch in test_loader:
        x, y = batch
        if isinstance(x, (list, tuple)):
            x = x[0]
        x, y = x.to(device), y.to(device)
        feats = extract_feats(model, x)

        protos = cil_method.prototypes(n_known)
        scores = osr_method.score(feats, protos)

        all_scores.append(scores.cpu())
        all_labels.append((y >= n_known).cpu().long())

    scores_t = torch.cat(all_scores).numpy()
    labels_t = torch.cat(all_labels).numpy()

    if len(np.unique(labels_t)) < 2:
        return 0.5, 1.0
    auroc = roc_auc_score(labels_t, scores_t)

    pos_scores = scores_t[labels_t == 1]
    neg_scores = scores_t[labels_t == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return auroc, 1.0
    thr = np.percentile(pos_scores, 5)
    fpr = (neg_scores >= thr).mean()
    return float(auroc), float(fpr)


# ======================================================================
# Training and Evaluation
# ======================================================================

def train_one_experiment(cil_name, osr_name, dataset, dataroot,
                         num_base, num_novel, num_all,
                         device, seed=3420, n_sessions=5,
                         log_dir: Optional[str] = None):
    """Train and evaluate one CIL × OSR combination on one dataset.

    Each CIL method owns its own encoder (standard ResNet18, not MYNET).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Experiment: {cil_name} × {osr_name} on {dataset}")
    print(f"Base={num_base}, Novel={num_novel}, Total={num_all}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Build dataset config (no MYNET model creation!)
    # ------------------------------------------------------------------
    LBRS, OpenDS = init_audio_dataset(dataset)

    class Args:
        pass
    args = Args()
    args.dataset = dataset
    args.dataroot = dataroot
    args.num_base = num_base
    args.num_novel = num_novel
    args.num_all = num_all
    args.way = 5
    args.n_ways = 5
    args.n_shots = 5
    args.n_queries = 15
    args.n_open_ways = 5
    args.train_episode = 50
    args.tmp_train = False
    args.test_times = 50
    args.feat_dim = 512
    args.num_labeled_classes = num_all
    args.seq_sample = False
    args.seed = seed
    args.lr_new = 0.1
    args.epochs = type('E', (), {'epochs_std': 30, 'epochs_new': 5})()
    args.lr = type('L', (), {'lr_std': 0.005, 'lr_new': 0.1, 'lrg': 0.1})()
    from collections import OrderedDict
    from addict import Dict
    args.optimizer = Dict({'decay': 5e-4, 'momentum': 0.9})
    args.scheduler = Dict({'schedule': 'Step', 'step': 40, 'gamma': 0.5})
    args.network = Dict({'temperature': 1, 'base_mode': 'ft_cos', 'new_mode': 'ft_cos'})
    args.episode = Dict({'train_episode': 50, 'low_way': 5, 'low_shot': 5, 'episode_way': 5, 'episode_shot': 5, 'episode_query': 15})
    args.stdu = Dict({'num_tmpb': 55, 'num_tmpi': 25, 'num_tmps': 14, 'num_incre': 5})
    args.dataloader = Dict({'num_workers': 2, 'train_batch_size': 128, 'test_batch_size': 100})
    args.extractor = Dict({'sample_rate': 16000, 'window_size': 400, 'hop_size': 160, 'mel_bins': 128, 'fmin': 0, 'fmax': 8000, 'window': 'hann'})
    args.train_weight_base = True
    args.Dataset = type('DS', (), {'LBRS': LBRS, 'Openlbrs': OpenDS,
                                   'NDS': LBRS, 'Opennds': OpenDS,
                                   'FSDCLIPS': LBRS, 'Openfs': OpenDS})()

    # ------------------------------------------------------------------
    # 2. Build base training dataloader
    # ------------------------------------------------------------------
    from data.dataloader import get_pretrain_dataloader
    trainset, base_loader = get_pretrain_dataloader(args)
    print(f"Base train loader: {len(base_loader)} batches")

    # ------------------------------------------------------------------
    # 3. Build CIL method — creates its own encoder internally
    # ------------------------------------------------------------------
    cil_method = build_cil(cil_name, args).to(device)
    osr_method = build_osr(osr_name, args)

    model = cil_method.model   # Each CIL method owns its model
    print(f"CIL: {type(cil_method).__name__}  |  "
          f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ------------------------------------------------------------------
    # 4. Session 0: Base training
    # ------------------------------------------------------------------
    print("\n--- Session 0: Base Training ---")
    if hasattr(cil_method, 'train_base') and cil_method.train_base is not None:
        cil_method.train_base(args, base_loader)
    else:
        from repro_baselines.methods.base import train_backbone_with_loss
        train_backbone_with_loss(
            model, args, base_loader,
            epochs=30, tag=f'{cil_name}_base')

    # ------------------------------------------------------------------
    # 5. Fit OSR methods that need training data statistics
    # ------------------------------------------------------------------
    if hasattr(osr_method, 'fit'):
        print("  Fitting OSR method on base training data...")
        osr_method.fit(model, base_loader)

    # ------------------------------------------------------------------
    # 6. Sessions 1-4: OSR → Clustering → Register → Evaluate
    # ------------------------------------------------------------------
    from sklearn.cluster import KMeans
    from utils.util import cluster_acc
    from data.dataloader import get_new_dataloader
    from data.dataloader import get_testloader
    from data.dataloader import get_inc_testloader

    all_results = {}

    # ---- Session 0 Evaluation ----
    session = 0
    n_known = num_base
    _, test_loader = get_testloader(args, session)
    all_acc = compute_acc(model, cil_method, test_loader, n_known, device)
    auroc, fpr95 = compute_osr_metrics(
        model, cil_method, osr_method, test_loader, num_base, device)
    all_results[session] = {
        'session': session, 'all_acc': all_acc, 'inc_acc': 0.0,
        'auroc': auroc, 'fpr95': fpr95,
    }
    print(f"  Session 0: all_acc={all_acc:.4f}  auroc={auroc:.4f}  "
          f"fpr95={fpr95:.4f}")

    # ---- Sessions 1-4 ----
    for session in range(1, n_sessions):
        print(f"\n--- Session {session} ---")

        # a. Load novel class support set (5-way, 5-shot)
        _, support_loader = get_new_dataloader(args, session)
        support_feats, support_labels = [], []
        for batch in support_loader:
            x, y = batch
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            with torch.no_grad():
                feats = extract_feats(model, x).detach()
            support_feats.append(feats.cpu())
            support_labels.append(y.cpu())
        support_feats = torch.cat(support_feats, 0)   # [N, 512]
        support_labels = torch.cat(support_labels, 0)  # [N]
        n_novel_cls = args.way  # 5

        # b. OSR scoring on support features
        n_known_before = num_base + (session - 1) * n_novel_cls
        protos_before = cil_method.prototypes(n_known_before)
        scores = osr_method.score(support_feats.to(device),
                                  protos_before.to(device))

        # c. Threshold → known/unknown split
        thr = torch.quantile(scores, 0.5)
        unk_mask = (scores >= thr).cpu()
        unk_feats = support_feats[unk_mask]
        unk_labs = support_labels[unk_mask]

        # d. KMeans clustering on unknown features
        cluster_accuracy = 0.0
        if unk_feats.shape[0] >= n_novel_cls:
            old_nlc = args.num_labeled_classes
            args.num_labeled_classes = n_known_before
            km = KMeans(n_clusters=n_novel_cls, n_init=20,
                        random_state=seed).fit(unk_feats.cpu().numpy())
            cluster_accuracy, mapping = cluster_acc(
                args, unk_labs.numpy(), km.labels_)
            args.num_labeled_classes = old_nlc

            # e. Register cluster prototypes as novel classes
            novel_ids, novel_protos = [], []
            for c in range(n_novel_cls):
                tgt = mapping.get(c, None)
                if tgt is None or tgt < n_known_before:
                    continue
                idx = torch.from_numpy(km.labels_ == c)
                if idx.sum() == 0:
                    continue
                proto = unk_feats[idx].mean(0)
                novel_ids.append(int(tgt))
                novel_protos.append(proto)

            if novel_protos:
                proto_bank = torch.stack(novel_protos, 0).to(device)
                cil_method.register_novel_classes(proto_bank, novel_ids)

        # f. Evaluate
        n_known = n_known_before + n_novel_cls
        _, test_loader = get_testloader(args, session)
        all_acc = compute_acc(model, cil_method, test_loader,
                              n_known, device)
        _, inc_test_loader = get_inc_testloader(args, session)
        inc_acc = compute_acc(model, cil_method, inc_test_loader,
                              n_known, device)
        auroc, fpr95 = compute_osr_metrics(
            model, cil_method, osr_method, test_loader, num_base, device)

        all_results[session] = {
            'session': session, 'all_acc': all_acc, 'inc_acc': inc_acc,
            'auroc': auroc, 'fpr95': fpr95,
            'cluster_acc': cluster_accuracy,
        }
        print(f"  all_acc={all_acc:.4f}  inc_acc={inc_acc:.4f}  "
              f"auroc={auroc:.4f}  fpr95={fpr95:.4f}  "
              f"cluster_acc={cluster_accuracy:.4f}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Summary: {cil_name} × {osr_name} on {dataset}")
    print(f"{'='*60}")
    session_accs = [all_results[s]['all_acc'] for s in range(n_sessions)]
    inc_accs = [all_results[s]['inc_acc'] for s in range(n_sessions) if s > 0]
    aurocs = [all_results[s]['auroc'] for s in range(n_sessions)]
    print(f"Session accs: {', '.join(f'{v:.4f}' for v in session_accs)}")
    print(f"AA_all: {np.mean(session_accs):.4f}")
    print(f"AA_inc: {np.mean(inc_accs):.4f}" if inc_accs else "AA_inc: N/A")
    print(f"PD_all: {session_accs[0] - session_accs[-1]:.4f}")
    print(f"AUROC S0: {aurocs[0]:.4f}  AUROC S4: {aurocs[-1]:.4f}")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        res_path = osp.join(log_dir, 'results.txt')
        with open(res_path, 'w') as fp:
            for s, r in all_results.items():
                fp.write(f"Session {s}: {r}\n")
            fp.write(f"\nAA_all: {np.mean(session_accs):.4f}\n")
            if inc_accs:
                fp.write(f"AA_inc: {np.mean(inc_accs):.4f}\n")
            fp.write(f"PD_all: {session_accs[0] - session_accs[-1]:.4f}\n")
        print(f"Results saved to {res_path}")

    return all_results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cil', type=str, required=True,
                        help='CIL method name')
    parser.add_argument('--osr', type=str, required=True,
                        help='OSR method name')
    parser.add_argument('--dataset', type=str, default='librispeech',
                        help='Dataset name')
    parser.add_argument('--dataroot', type=str,
                        default='/data/datasets/librispeech_fscil/',
                        help='Data root')
    parser.add_argument('--num_base', type=int, default=80)
    parser.add_argument('--num_novel', type=int, default=20)
    parser.add_argument('--num_all', type=int, default=100)
    parser.add_argument('--seed', type=int, default=3420)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = osp.join(ROOT, 'repro_baselines', 'logs',
                           args.dataset, f'{args.cil}_{args.osr}')

    train_one_experiment(
        cil_name=args.cil,
        osr_name=args.osr,
        dataset=args.dataset,
        dataroot=args.dataroot,
        num_base=args.num_base,
        num_novel=args.num_novel,
        num_all=args.num_all,
        device=device,
        seed=args.seed,
        n_sessions=5,
        log_dir=log_dir,
    )


if __name__ == '__main__':
    main()
