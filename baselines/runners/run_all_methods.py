#!/usr/bin/env python3
"""
Unified runner for all OSR baseline methods across datasets.

Optimized: pre-extracts features once, then evaluates all methods.

Usage:
    python -m baselines.runners.run_all_methods --dataset ls100
    python -m baselines.runners.run_all_methods --dataset ns100
    python -m baselines.runners.run_all_methods --dataset ls100 --methods energy mahalanobis
    python -m baselines.runners.run_all_methods --all

Output:
    - baselines/tables/{method}_{dataset}.json     per-method results
    - baselines/tables/summary_{dataset}.json       combined summary
    - baselines/tables/comparison_{dataset}.md       markdown table
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_BASELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASELINE_ROOT)

from baselines.methods import build_osr
from baselines.eval.metrics import compute_per_sample_metrics, compute_auroc_fpr95
from baselines.eval.reporter import generate_main_table, generate_summary_json


# ============================================================
# Config
# ============================================================
DATASET_CONFIGS = {
    'ls100': {
        'dataset': 'librispeech',
        'dataroot': '/data/datasets/librispeech_fscil/',
        'num_base': 80,
        'num_novel': 20,
        'way': 5,
        'n_shot': 5,
        'pretrained_model': '/data/lqq/baseline/save/base_train_for_meta_ls.pth',
        'sample_rate': 16000,
        'n_fft': 400,
        'hop_size': 160,
        'mel_bins': 128,
        'batch_size': 128,
        'test_batch_size': 100,
    },
    'ns100': {
        'dataset': 'nsynth',
        'dataroot': '/data/datasets/The_NSynth_Dataset',
        'num_base': 80,
        'num_novel': 20,
        'way': 5,
        'n_shot': 5,
        'pretrained_model': '/data/lqq/baseline/save/base_train_for_meta_ns.pth',
        'sample_rate': 16000,
        'n_fft': 400,
        'hop_size': 160,
        'mel_bins': 128,
        'batch_size': 128,
        'test_batch_size': 100,
    },
}

ALL_METHODS = ['proto', 'energy', 'mahalanobis', 'openmax', 'dnpg', 'foac_aifp', 'pclae_ctpn']

# ============================================================
# Model loader
# ============================================================
class DotDict(dict):
    def __getattr__(self, key):
        if key in self:
            v = self[key]
            if isinstance(v, dict):
                return DotDict(v)
            return v
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")
    def __setattr__(self, key, value):
        self[key] = value


def make_mynet_args(config):
    a = DotDict()
    a.extractor = DotDict({
        'sample_rate': config['sample_rate'],
        'window_size': config['n_fft'],
        'hop_size': config['hop_size'],
        'mel_bins': config['mel_bins'],
        'fmin': 0, 'fmax': 8000, 'window': 'hann',
    })
    a.network = DotDict({'temperature': 10.0})
    a.train_weight_base = False
    a.n_ways = 5; a.n_shots = 5; a.n_open_ways = 5; a.n_queries = 15
    a.base_seman_calib = False; a.neg_gen_type = 'att'; a.agg = 'mean'
    a.hinge_margin = 2.0; a.dual_cos_weight = 0.0; a.dual_cos_margin = 0.2
    a.neg_div_margin = 0.1; a.inter_cos_weight = 0.0; a.inter_cos_margin = 0.3
    a.osr_margin_weight = 0.0; a.osr_margin_val = 0.5
    a.use_uop = False; a.uop_chunk_size = 5; a.uop_mode = 'antimean'
    a.osr_noise_std = 0.1
    a.num_labeled_classes = config['num_base']
    a.tmp_train = False
    a.num_base = config['num_base']
    a.num_all = 100; a.way = config['way']; a.num_novel = config['num_novel']
    a.test_times = 50; a.n_test_runs = 100; a.n_train_runs = 100
    a.seq_sample = False
    a.strategy = DotDict({'data_init': True})
    a.episode = DotDict({'train_episode': 50, 'episode_way': 5, 'episode_shot': 5, 'episode_query': 15, 'low_way': 5, 'low_shot': 5})
    a.lr = DotDict({'lr_std': 0.005, 'lrg': 0.1, 'lr_new': 0.1, 'lr_decay_rate': 0.1})
    a.scheduler = DotDict({'schedule': 'Step', 'step': 40, 'gamma': 0.5, 'milestones': [40, 80]})
    a.optimizer = DotDict({'decay': 0.0005, 'momentum': 0.9})
    a.stdu = DotDict({'num_tmpb': 55, 'num_tmpi': 25, 'num_tmps': 14, 'num_incre': 5, 'pqa': False})
    a.dataloader = DotDict({'num_workers': 4, 'train_batch_size': 128, 'test_batch_size': 100})
    a.epochs = DotDict({'epochs_std': 30, 'epochs_meta': 15, 'epochs_stdu_base': 1, 'epochs_new': 5})
    return a


def load_model(config, device):
    from network import MYNET
    mynet_args = make_mynet_args(config)
    model = MYNET(mynet_args, mode='encoder')
    pretrained_path = config['pretrained_model']
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state = ckpt.get('params', ckpt)
        if any(k.startswith('module.') for k in state):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"  Loaded pretrained model: {pretrained_path}")
    else:
        print(f"  WARNING: pretrained not found at {pretrained_path}")
    model = model.to(device)
    model.mode = 'incre'  # Ensure encode returns 512-dim features, not logits
    model.eval()
    return model


# ============================================================
# Data helpers
# ============================================================
def get_base_val_data(config):
    """Get base class validation samples for calibration."""
    dataset_name = config['dataset']
    dataroot = config['dataroot']
    num_base = config['num_base']
    base_classes = np.arange(num_base)

    if 'librispeech' in dataset_name:
        from data import librispeech
        ds = librispeech.LBRS(root=dataroot, phase='val', index=base_classes, k=None, base_sess=True)
    elif 'nsynth' in dataset_name:
        from data import nsynth
        ds = nsynth.NDS(root=dataroot, phase='val', index=base_classes, k=None, base_sess=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return ds


def get_session_test_loader(config, session):
    """Test loader for classes 0 .. num_base + session*way."""
    dataset_name = config['dataset']
    dataroot = config['dataroot']
    num_base = config['num_base']
    way = config['way']
    end_cls = num_base + session * way
    class_new = np.arange(0, end_cls)
    batch_size = config['test_batch_size']

    if 'librispeech' in dataset_name:
        from data import librispeech
        ds = librispeech.LBRS(root=dataroot, phase='test', index=class_new, k=None)
    elif 'nsynth' in dataset_name:
        from data import nsynth
        ds = nsynth.NDS(root=dataroot, phase='test', index=class_new, k=None)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


def extract_features(model, dataset, indices, batch_size, device):
    """Extract features from specific indices."""
    model.eval()
    feats, labels = [], []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        batch_data = []
        batch_labels = []
        max_len = 0
        for idx in batch_idx:
            data, label = dataset[idx]
            if isinstance(data, torch.Tensor):
                if data.dim() == 1:
                    data = data.unsqueeze(0)  # [1, T]
                elif data.dim() == 2 and data.size(0) > 1:
                    data = data[0:1, :]  # take first channel
            else:
                data = torch.tensor(data).unsqueeze(0)
            max_len = max(max_len, data.shape[-1])
            batch_data.append(data)
            batch_labels.append(label)
        padded = []
        for d in batch_data:
            if d.shape[-1] < max_len:
                d = F.pad(d, (0, max_len - d.shape[-1]))
            # Ensure shape is [1, T]
            if d.dim() == 1:
                d = d.unsqueeze(0)
            padded.append(d)
        batch_tensor = torch.cat(padded, dim=0).to(device)  # [B, T]
        with torch.no_grad():
            batch_feat = model.encode(batch_tensor)
        feats.append(batch_feat.cpu())
        labels.extend(batch_labels)
    return torch.cat(feats, dim=0), torch.tensor(labels)


# ============================================================
# Feature cache
# ============================================================
_feature_cache = {}

def get_all_test_features(config, model, device):
    """Pre-extract features for ALL test classes (0-99).
    
    Returns:
        features: [N, D] all test features
        labels: [N] all test labels
    """
    cache_key = config['dataset']
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]
    
    dataset_name = config['dataset']
    dataroot = config['dataroot']
    all_classes = np.arange(0, 100)
    
    print("  Loading test dataset...")
    if 'librispeech' in dataset_name:
        from data import librispeech
        ds = librispeech.LBRS(root=dataroot, phase='test', index=all_classes, k=None)
    elif 'nsynth' in dataset_name:
        from data import nsynth
        ds = nsynth.NDS(root=dataroot, phase='test', index=all_classes, k=None)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"  Extracting features for {len(ds)} samples...")
    targets = np.array(ds.targets)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['test_batch_size'], 
                                          shuffle=False, num_workers=0, pin_memory=True)
    
    feats_list, labels_list = [], []
    model.eval()
    for data, labels in tqdm(loader, desc='  FeatExtract'):
        data = data.to(device)
        with torch.no_grad():
            f = model.encode(data)
        feats_list.append(f.cpu())
        labels_list.append(labels)
    
    features = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    _feature_cache[cache_key] = (features, labels)
    print(f"  Pre-extracted features: {features.shape}")
    return features, labels


def get_calibration_features(config, model, device):
    """Get calibration features from base class validation set."""
    dataset_name = config['dataset']
    dataroot = config['dataroot']
    num_base = config['num_base']
    base_classes = np.arange(num_base)
    
    if 'librispeech' in dataset_name:
        from data import librispeech
        ds = librispeech.LBRS(root=dataroot, phase='val', index=base_classes, k=None, base_sess=True)
    elif 'nsynth' in dataset_name:
        from data import nsynth
        ds = nsynth.NDS(root=dataroot, phase='val', index=base_classes, k=None, base_sess=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    targets = np.array(ds.targets)
    # Sample up to 2000 evenly
    calib_indices = []
    per_class = max(1, min(2000 // num_base, 30))
    for c in range(num_base):
        cls_idx = np.where(targets == c)[0]
        n = min(per_class, len(cls_idx))
        calib_indices.extend(np.random.choice(cls_idx, n, replace=False).tolist())
    
    # Extract features for these indices using DataLoader
    # Build subset
    sub_data, sub_labels = [], []
    for idx in calib_indices:
        d, l = ds[idx]
        sub_data.append(d)
        sub_labels.append(l)
    
    # Manual batching
    feats = []
    batch_size = 64
    for i in range(0, len(sub_data), batch_size):
        batch = sub_data[i:i+batch_size]
        max_len = max(d.shape[-1] if d.dim() >= 1 else 0 for d in batch)
        padded = []
        for d in batch:
            if d.dim() == 1:
                d = d.unsqueeze(0)
            if d.shape[-1] < max_len:
                d = F.pad(d, (0, max_len - d.shape[-1]))
            padded.append(d)
        batch_tensor = torch.cat(padded, dim=0).to(device)
        with torch.no_grad():
            f = model.encode(batch_tensor)
        feats.append(f.cpu())
    
    features = torch.cat(feats, dim=0)
    labels = torch.tensor(sub_labels)
    return features, labels


# ============================================================
# Main pipeline (optimized: uses pre-extracted features)
# ============================================================
def run_method(config, method_name, device, output_dir, all_feats, all_labels):
    """Run full pipeline for one OSR method on one dataset."""
    dataset_name = config['dataset']
    num_base = config['num_base']
    way = config['way']
    num_sessions = config['num_novel'] // way

    print(f"\n  Method: {method_name}  |  Dataset: {dataset_name}")
    print(f"  {'-'*50}")

    t0 = time.time()
    
    # 1. Load model (need fc.weight for prototypes)
    model = load_model(config, device)
    base_protos = model.fc.weight.data[:num_base].clone()
    
    # 2. Calibrate OSR
    calib_feats, calib_labels = get_calibration_features(config, model, device)
    calib_feats = calib_feats.to(device)
    calib_labels = calib_labels.to(device)
    if method_name == 'proto':
        osr = build_osr('proto', temperature=10.0)
        osr.calibrate(calib_feats, base_protos, known_pct=0.95)
    elif method_name == 'energy':
        osr = build_osr('energy', temperature=1.0)
        osr.calibrate(calib_feats, base_protos, known_pct=0.95)
    elif method_name == 'mahalanobis':
        osr = build_osr('mahalanobis', eps=0.001)
        osr.calibrate(calib_feats, calib_labels, known_pct=0.95)
    elif method_name == 'openmax':
        osr = build_osr('openmax', tailsize=20, alpha=3, distance_type='euclidean')
        osr.calibrate(calib_feats, calib_labels, known_pct=0.95)
    elif method_name == 'dnpg':
        osr = build_osr('dnpg', n_neg_per_class=3, noise_scale=0.3, temperature=10.0)
        osr.calibrate(calib_feats, base_protos, calib_labels, known_pct=0.95)
    elif method_name == 'foac_aifp':
        osr = build_osr('foac_aifp', temperature=10.0, dim=512)
        osr.fusion = osr.fusion.to(device)
        osr.calibrate(calib_feats, base_protos, calib_labels, known_pct=0.95)
    elif method_name == 'pclae_ctpn':
        osr = build_osr('pclae_ctpn', radius_percentile=0.90, temperature=10.0)
        osr.calibrate(calib_feats, base_protos, calib_labels, known_pct=0.95)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # 3. Session 0: base evaluation (classes 0-79)
    session0_mask = all_labels < num_base
    s0_feats = all_feats[session0_mask]
    s0_labels = all_labels[session0_mask]
    session0_acc = eval_cosine_features(s0_feats, s0_labels, base_protos, device, num_base)
    
    # 4. Incremental sessions
    current_protos = base_protos.clone()
    num_known = num_base
    all_results = []
    T = 10.0
    
    # Pre-compute novel class prototypes from train set
    novel_protos_cache = _precompute_novel_protos(config, model, device)
    
    for session in range(1, num_sessions + 1):
        # Classes 0 .. num_base + session*way
        end_cls = num_base + session * way
        session_mask = all_labels < end_cls
        sess_feats = all_feats[session_mask].to(device)
        sess_labels = all_labels[session_mask]
        
        # Move prototypes to same device
        cur_proto = current_protos.to(device)
        
        # OSR detection — get method-specific scores
        is_unk_gt = (sess_labels >= num_known).to(device)
        
        if method_name in ('mahalanobis', 'dnpg', 'foac_aifp', 'pclae_ctpn'):
            osr_scores = osr.score(sess_feats)  # higher = more unknown
            known_mask = osr_scores < osr.threshold
        else:
            osr_scores = osr.score(sess_feats, cur_proto)  # higher = more unknown
            known_mask = osr_scores < osr.threshold
        
        is_unk_pred = (~known_mask).cpu().numpy()
        
        # Classification
        feat_n = F.normalize(sess_feats, dim=1)
        proto_n = F.normalize(cur_proto, dim=1)
        cos_sim = (feat_n @ proto_n.T) * T
        pred_cls = cos_sim.argmax(dim=1).cpu().numpy()
        pred_cls[is_unk_pred] = -1
        
        # Use OSR method scores for AUROC (not classifier scores)
        osr_scores_cpu = osr_scores.detach().cpu()
        is_unk_gt_cpu = is_unk_gt.cpu()
        all_results.append({
            'gt': sess_labels.cpu().numpy(),
            'pred': pred_cls,
            'is_unknown_gt': is_unk_gt_cpu.numpy(),
            'is_unknown_pred': is_unk_pred,
            'known_scores': osr_scores_cpu[~is_unk_gt_cpu].numpy().tolist(),
            'unknown_scores': osr_scores_cpu[is_unk_gt_cpu].numpy().tolist(),
        })
        
        # Add novel prototypes
        new_protos = novel_protos_cache[session - 1]  # [way, D]
        current_protos = torch.cat([current_protos.cpu(), new_protos], dim=0)
        num_known += way

    # 5. Metrics
    session_classes = [np.arange(0, num_base + (s + 1) * way) for s in range(num_sessions)]
    
    metrics = compute_per_sample_metrics(
        [r['gt'] for r in all_results],
        [r['pred'] for r in all_results],
        [r['is_unknown_gt'] for r in all_results],
        [r['is_unknown_pred'] for r in all_results],
        (0, num_base), session_classes
    )
    
    all_ks = []; all_us = []
    for r in all_results:
        all_ks.extend(r['known_scores'])
        all_us.extend(r['unknown_scores'])
    if len(all_ks) > 0 and len(all_us) > 0:
        auroc, fpr95 = compute_auroc_fpr95(all_ks, all_us, np.zeros(len(all_ks)), np.ones(len(all_us)))
    else:
        auroc, fpr95 = 0.5, 1.0
    
    metrics['AUROC'] = float(auroc)
    metrics['FPR95'] = float(fpr95)
    metrics['session0_acc'] = float(session0_acc)
    metrics['method'] = method_name
    metrics['dataset'] = dataset_name
    
    elapsed = time.time() - t0
    print(f"    S0={metrics['session0_acc']:.4f} | AA_inc={metrics['AA_inc']:.4f} | AA_all={metrics['AA_all']:.4f} | "
          f"AUROC={metrics['AUROC']:.4f} | FPR95={metrics['FPR95']:.4f} | [{elapsed:.1f}s]")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{method_name}_{dataset_name}.json')
    with open(out_path, 'w') as f:
        json.dump({k: v for k, v in metrics.items() if not isinstance(v, (np.ndarray,))}, f, indent=2)
    
    return metrics


def _precompute_novel_protos(config, model, device):
    """Pre-compute novel class prototypes from train set (5-shot)."""
    dataset_name = config['dataset']
    dataroot = config['dataroot']
    num_base = config['num_base']
    way = config['way']
    n_shot = config['n_shot']
    num_sessions = config['num_novel'] // way
    
    novel_protos_per_session = []
    for session in range(1, num_sessions + 1):
        new_classes = np.arange(num_base + (session - 1) * way, num_base + session * way)
        if 'librispeech' in dataset_name:
            from data import librispeech
            ds = librispeech.LBRS(root=dataroot, phase='train', index=new_classes, k=n_shot, base_sess=True)
        elif 'nsynth' in dataset_name:
            from data import nsynth
            ds = nsynth.NDS(root=dataroot, phase='train', index=new_classes, k=n_shot, base_sess=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Extract features
        targets = np.array(ds.targets)
        all_data = [ds[i][0] for i in range(len(ds))]
        all_lbls = [ds[i][1] for i in range(len(ds))]
        
        feats = []
        for i in range(0, len(all_data), 32):
            batch = all_data[i:i+32]
            max_len = max(d.shape[-1] if d.dim() >= 1 else 0 for d in batch)
            padded = []
            for d in batch:
                if d.dim() == 1: d = d.unsqueeze(0)
                if d.shape[-1] < max_len: d = F.pad(d, (0, max_len - d.shape[-1]))
                padded.append(d)
            batch_tensor = torch.cat(padded, dim=0).to(device)
            with torch.no_grad():
                f = model.encode(batch_tensor)
            feats.append(f.cpu())
        feats = torch.cat(feats, dim=0)
        lbls = torch.tensor(all_lbls)
        
        protos = []
        for c in new_classes:
            mask = lbls == c
            if mask.sum() > 0:
                protos.append(feats[mask].mean(dim=0))
            else:
                protos.append(torch.zeros(512))
        novel_protos_per_session.append(torch.stack(protos, dim=0))
    
    return novel_protos_per_session


def eval_cosine_features(feats, labels, prototypes, device, n_class):
    """Evaluate accuracy using pre-extracted features."""
    proto = prototypes.to(device)
    T = 10.0
    feat_n = F.normalize(feats.to(device), dim=1)
    proto_n = F.normalize(proto[:n_class], dim=1)
    logits = (feat_n @ proto_n.T) * T
    pred = logits.argmax(dim=1).cpu()
    labels = labels.cpu()
    correct = (pred == labels).sum().item()
    return correct / max(len(labels), 1)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Run all OSR baselines')
    parser.add_argument('--dataset', type=str, default='ls100',
                        choices=['ls100', 'ns100'], help='Dataset to use')
    parser.add_argument('--methods', nargs='+', default=None,
                        help=f'Methods to run. Available: {ALL_METHODS}')
    parser.add_argument('--all', action='store_true', help='Run ALL methods')
    parser.add_argument('--output_dir', type=str, default='baselines/tables')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_test_runs', type=int, default=1,
                        help='Number of test runs per method (for averaging)')
    args_cli = parser.parse_args()

    device = torch.device(args_cli.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = dict(DATASET_CONFIGS[args_cli.dataset])
    config['dataset_name'] = args_cli.dataset

    if args_cli.all:
        methods = list(ALL_METHODS)
    elif args_cli.methods:
        methods = [m for m in args_cli.methods if m in ALL_METHODS]
    else:
        methods = list(ALL_METHODS)

    print(f"Dataset: {args_cli.dataset} ({config['dataset']})")
    print(f"Methods: {methods}")

    # ===== Pre-extract all test features (shared across methods) =====
    print("\n=== Pre-extracting test features (one-time) ===")
    extract_model = load_model(config, device)
    all_feats, all_labels = get_all_test_features(config, extract_model, device)
    print(f"  Test features: {all_feats.shape}, labels: {all_labels.shape}")
    # Free model memory for subsequent method runs
    del extract_model
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    # ===== Run each method =====
    all_metrics = {}
    for method in methods:
        t0 = time.time()
        run_metrics = []
        for run in range(args_cli.n_test_runs):
            m = run_method(config, method, device, args_cli.output_dir, all_feats, all_labels)
            run_metrics.append(m)
        # Average
        avg = {}
        for key in ['session0_acc', 'AA_inc', 'AA_all', 'AA_known', 'AA_unknown',
                     'PD_inc', 'PD_all', 'AUROC', 'FPR95']:
            vals = [rm[key] for rm in run_metrics if key in rm]
            avg[key] = float(np.mean(vals)) if vals else 0.0
        if 'per_session' in run_metrics[0]:
            n_sessions = len(run_metrics[0]['per_session'].get('inc_acc', []))
            avg['per_session'] = {}
            for k in ['inc_acc', 'all_acc', 'acc_known', 'acc_unknown']:
                arr = np.array([rm['per_session'].get(k, [0]*n_sessions) for rm in run_metrics])
                avg['per_session'][k] = arr.mean(axis=0).tolist()
        avg['method'] = method
        avg['dataset'] = config['dataset']
        avg['dataset_name'] = args_cli.dataset
        all_metrics[method] = avg
        print(f"  [{method}] Done in {time.time()-t0:.1f}s | AA_all={avg['AA_all']:.4f} AUROC={avg['AUROC']:.4f}\n")

    # Save combined
    os.makedirs(args_cli.output_dir, exist_ok=True)
    summary_path = os.path.join(args_cli.output_dir, f'summary_{args_cli.dataset}.json')
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Generate markdown table
    _results_for_table = {m: {config['dataset']: v} for m, v in all_metrics.items()}
    table_path = os.path.join(args_cli.output_dir, f'comparison_{args_cli.dataset}.md')
    generate_main_table(_results_for_table, table_path)

    # Quick summary
    print(f"\n{'='*100}")
    print(f"  Results: {args_cli.dataset}")
    print(f"{'='*100}")
    header = f"{'Method':<16} {'S0_acc':>8} {'AA_inc':>8} {'AA_all':>8} {'AA_known':>8} {'AA_unknown':>8} {'AUROC':>8} {'FPR95':>8}"
    print(header)
    print("-" * len(header))
    for method in methods:
        m = all_metrics[method]
        print(f"{method:<16} {m['session0_acc']:>8.4f} {m['AA_inc']:>8.4f} {m['AA_all']:>8.4f} "
              f"{m['AA_known']:>8.4f} {m['AA_unknown']:>8.4f} {m['AUROC']:>8.4f} {m['FPR95']:>8.4f}")


if __name__ == '__main__':
    main()
