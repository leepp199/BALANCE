"""
Main pipeline for FOWAC baseline reproduction.
Supports multiple OSR methods: energy, mahalanobis, openmax, proto.
Architecture: Load pretrained MYNET encoder -> {OSR detect -> Prototype update} x Sessions.

Uses existing pretrained base models so that all OSR baselines share the same encoder.
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict
import json

_BASELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _BASELINE_ROOT)
from data import librispeech, nsynth
from network import MYNET
from baselines.eval.metrics import compute_per_sample_metrics, compute_auroc_fpr95


# ============================================================
# Args helper for MYNET
# ============================================================
class DotDict(dict):
    """Dict with dot-notation access for nested keys."""
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
    """Build an args object compatible with MYNET.__init__."""
    a = DotDict()
    
    # Extractor settings
    a.extractor = DotDict({
        'sample_rate': config.get('sample_rate', 16000),
        'window_size': config.get('n_fft', 400),
        'hop_size': config.get('hop_size', 160),
        'mel_bins': config.get('mel_bins', 128),
        'fmin': 0,
        'fmax': 8000,
        'window': 'hann',
    })
    
    # Network settings
    a.network = DotDict({'temperature': config.get('temperature', 10.0)})
    
    # Episode / meta-training settings
    a.train_weight_base = False
    a.n_ways = 5
    a.n_shots = 5
    a.n_open_ways = 5
    a.n_queries = 15
    
    # AttnClassifier settings
    a.base_seman_calib = False
    a.neg_gen_type = 'att'
    a.agg = 'mean'
    
    # OSR settings
    a.hinge_margin = 2.0
    a.dual_cos_weight = 0.0
    a.dual_cos_margin = 0.2
    a.neg_div_margin = 0.1
    a.inter_cos_weight = 0.0
    a.inter_cos_margin = 0.3
    a.osr_margin_weight = 0.0
    a.osr_margin_val = 0.5
    a.use_uop = False
    a.uop_chunk_size = 5
    a.uop_mode = 'antimean'
    a.osr_noise_std = 0.1
    
    # Incremental / session settings
    a.num_labeled_classes = config['num_base']
    a.tmp_train = False
    a.num_base = config['num_base']
    a.num_all = config.get('num_all', 100)
    a.way = config.get('way', 5)
    a.num_novel = config.get('num_novel', 20)
    a.test_times = 50
    a.n_test_runs = 100
    a.n_train_runs = 100
    a.seq_sample = False
    a.strategy = DotDict({'data_init': True})
    a.episode = DotDict({
        'train_episode': 50,
        'episode_way': 5,
        'episode_shot': 5,
        'episode_query': 15,
        'low_way': 5,
        'low_shot': 5,
    })
    a.lr = DotDict({'lr_std': 0.005, 'lrg': 0.1, 'lr_new': 0.1, 'lr_decay_rate': 0.1})
    a.scheduler = DotDict({'schedule': 'Step', 'step': 40, 'gamma': 0.5, 'milestones': [40, 80]})
    a.optimizer = DotDict({'decay': 0.0005, 'momentum': 0.9})
    a.stdu = DotDict({'num_tmpb': 55, 'num_tmpi': 25, 'num_tmps': 14, 'num_incre': 5, 'pqa': False})
    a.dataloader = DotDict({'num_workers': 8, 'train_batch_size': 128, 'test_batch_size': 100})
    a.epochs = DotDict({'epochs_std': 30, 'epochs_meta': 15, 'epochs_stdu_base': 1, 'epochs_new': 5})
    
    return a


# ============================================================
# Dataset Helpers
# ============================================================
def get_base_loaders(args_dict):
    """Get base class train/val dataset refs for calibration (not DataLoaders)."""
    dataset_name = args_dict['dataset']
    dataroot = args_dict['dataroot']
    num_base = args_dict['num_base']
    base_classes = np.arange(num_base)
    
    if 'librispeech' in dataset_name:
        train_ds = librispeech.LBRS(root=dataroot, phase='train', index=base_classes, base_sess=True)
        val_ds = librispeech.LBRS(root=dataroot, phase='val', index=base_classes, k=None, base_sess=True)
    elif 'nsynth' in dataset_name:
        train_ds = nsynth.NDS(root=dataroot, phase='train', index=base_classes, base_sess=True)
        val_ds = nsynth.NDS(root=dataroot, phase='val', index=base_classes, k=None, base_sess=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_ds, val_ds


def get_session_test_loader(args_dict, session):
    """Get test loader for all classes encountered up to session."""
    dataset_name = args_dict['dataset']
    dataroot = args_dict['dataroot']
    num_base = args_dict['num_base']
    way = args_dict.get('way', 5)
    
    end_cls = num_base + session * way
    class_new = np.arange(0, end_cls)
    batch_size = args_dict.get('test_batch_size', 100)
    
    if 'librispeech' in dataset_name:
        testset = librispeech.LBRS(root=dataroot, phase='test', index=class_new, k=None)
    elif 'nsynth' in dataset_name:
        testset = nsynth.NDS(root=dataroot, phase='test', index=class_new, k=None)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def get_new_class_train_set(args_dict, session):
    """Get new class training set for prototype computation."""
    dataset_name = args_dict['dataset']
    dataroot = args_dict['dataroot']
    num_base = args_dict['num_base']
    way = args_dict.get('way', 5)
    
    new_classes = np.arange(num_base + (session - 1) * way, num_base + session * way)
    
    if 'librispeech' in dataset_name:
        return new_classes, librispeech.LBRS(root=dataroot, phase='train', index=new_classes, k=None, base_sess=True)
    elif 'nsynth' in dataset_name:
        return new_classes, nsynth.NDS(root=dataroot, phase='train', index=new_classes, k=None, base_sess=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================
# Utility: batch feature extraction without DataLoader
# ============================================================
def extract_features_batched(model, dataset, indices, batch_size, device='cuda'):
    """Extract features from a subset of samples using direct iteration."""
    model.eval()
    feats = []
    labels = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = []
        batch_labels = []
        max_len = 0
        for idx in batch_indices:
            data, label = dataset[idx]
            if isinstance(data, torch.Tensor):
                if data.dim() == 1:
                    data = data.unsqueeze(0)  # (T,) -> (1, T)
                batch_data.append(data)
            else:
                data = torch.tensor(data).unsqueeze(0)
                batch_data.append(data)
            max_len = max(max_len, data.shape[-1])
            batch_labels.append(label)
        # Pad all to max_len
        padded = []
        for d in batch_data:
            if d.shape[-1] < max_len:
                d = F.pad(d, (0, max_len - d.shape[-1]))
            padded.append(d)
        batch_tensor = torch.stack(padded).to(device)
        with torch.no_grad():
            batch_feat = model.encode(batch_tensor)
        feats.append(batch_feat.cpu())
        labels.extend(batch_labels)
    return torch.cat(feats, dim=0), torch.tensor(labels)


def calibrate_fast(model, fc_weights, dataset, num_base, osr_type, args, device='cuda'):
    """Fast calibration using a subsample of validation data.
    Uses at most calib_samples samples."""
    calib_samples = min(args.get('calib_samples', 1000), len(dataset))
    # Sample evenly from each class
    per_class = max(1, calib_samples // num_base)
    indices = []
    targets = np.array(dataset.targets)
    for c in range(num_base):
        cls_idx = np.where(targets == c)[0]
        n = min(per_class, len(cls_idx))
        indices.extend(np.random.choice(cls_idx, n, replace=False).tolist())
    
    print(f"  Calibrating on {len(indices)} samples...")
    feats, labels = extract_features_batched(model, dataset, indices, args.get('calib_batch', 64), device)
    
    if osr_type == 'proto':
        T = args.get('temperature', 10.0)
        proto_n = F.normalize(fc_weights.to(device), dim=1)
        feat_n = F.normalize(feats.to(device), dim=1)
        scores = (feat_n @ proto_n.T).max(dim=1)[0].cpu() * T
        threshold = float(torch.quantile(scores, 1.0 - args.get('osr_calib_pct', 0.95)))
        return threshold
    elif osr_type == 'energy':
        T = args.get('temperature', 1.0)
        logits = F.linear(feats.to(device), fc_weights.to(device)) / T
        energy = -T * torch.logsumexp(logits / T, dim=1).cpu()
        threshold = float(torch.quantile(energy, args.get('osr_calib_pct', 0.95)))
        return threshold
    elif osr_type == 'mahalanobis':
        # Compute per-class means and shared covariance
        feats_by_class = {}
        for c in range(num_base):
            mask = labels == c
            if mask.sum() > 0:
                feats_by_class[c] = feats[mask]
        means = torch.stack([feats_by_class[c].mean(0) for c in sorted(feats_by_class.keys())]).to(device)
        all_centered = torch.cat([feats_by_class[c] - feats_by_class[c].mean(0) for c in sorted(feats_by_class.keys())])
        cov = (all_centered.T @ all_centered) / max(len(all_centered) - 1, 1)
        cov += 0.001 * torch.eye(cov.shape[0])
        inv_cov = torch.linalg.pinv(cov).to(device)
        diff = feats.to(device).unsqueeze(1) - means.unsqueeze(0)
        dists = (diff @ inv_cov * diff).sum(dim=2).min(dim=1)[0].cpu()
        threshold = float(torch.quantile(dists, args.get('osr_calib_pct', 0.95)))
        return threshold, means, inv_cov
    else:
        return 0.5  # default
class EnergyOSR:
    """Energy-based OSR: lower energy = more likely OOD."""
    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self.threshold = None

    def calibrate(self, model, fc_weights, val_loader, known_pct=0.95, device='cuda'):
        with torch.no_grad():
            model.eval()
            scores = []
            for data, _ in val_loader:
                feat = model.encode(data.to(device))
                logits = F.linear(feat, fc_weights) / self.temperature
                energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
                scores.extend(energy.cpu().numpy())
        self.threshold = float(np.percentile(np.array(scores), known_pct * 100))
        return self.threshold

    def get_scores(self, model, fc_weights, data, device='cuda'):
        with torch.no_grad():
            model.eval()
            feat = model.encode(data.to(device))
            logits = F.linear(feat, fc_weights) / self.temperature
            return -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)


class MahalanobisOSR:
    """Mahalanobis distance OSR."""
    def __init__(self):
        self.means = None
        self.inv_cov = None
        self.threshold = None

    def calibrate(self, model, fc_weights, val_loader, known_pct=0.95, device='cuda'):
        with torch.no_grad():
            model.eval()
            feats_by_class = defaultdict(list)
            for data, labels in val_loader:
                feat = model.encode(data.to(device))
                for f, l in zip(feat.cpu(), labels):
                    feats_by_class[l.item()].append(f)
            
            self.means = []
            all_centered = []
            for cls in sorted(feats_by_class.keys()):
                cls_feats = torch.stack(feats_by_class[cls])
                mean = cls_feats.mean(dim=0)
                self.means.append(mean)
                all_centered.append(cls_feats - mean)
            
            self.means = torch.stack(self.means).to(device)
            all_centered = torch.cat(all_centered)
            cov = (all_centered.T @ all_centered) / max(len(all_centered) - 1, 1)
            cov += 0.001 * torch.eye(cov.shape[0])
            self.inv_cov = torch.linalg.pinv(cov).to(device)
            
            scores = []
            for data, _ in val_loader:
                feat = model.encode(data.to(device))
                diff = feat.unsqueeze(1) - self.means.unsqueeze(0)
                dists = (diff @ self.inv_cov * diff).sum(dim=2)
                scores.extend(dists.min(dim=1)[0].cpu().numpy())
        self.threshold = float(np.percentile(np.array(scores), known_pct * 100))
        return self.threshold

    def get_scores(self, model, fc_weights, data, device='cuda'):
        with torch.no_grad():
            model.eval()
            feat = model.encode(data.to(device))
            diff = feat.unsqueeze(1) - self.means.unsqueeze(0)
            dists = (diff @ self.inv_cov * diff).sum(dim=2)
            return dists.min(dim=1)[0]


class ProtoOSR:
    """Prototype-based OSR: max cosine similarity threshold."""
    def __init__(self, temperature=10.0):
        self.temperature = temperature
        self.threshold = None

    def calibrate(self, model, fc_weights, val_loader, known_pct=0.95, device='cuda'):
        with torch.no_grad():
            model.eval()
            scores = []
            proto_norm = F.normalize(fc_weights.to(device), dim=1)
            for data, _ in val_loader:
                feat = F.normalize(model.encode(data.to(device)), dim=1)
                max_sim = (feat @ proto_norm.T).max(dim=1)[0] * self.temperature
                scores.extend(max_sim.cpu().numpy())
        self.threshold = float(np.percentile(np.array(scores), (1 - known_pct) * 100))
        return self.threshold

    def get_scores(self, model, fc_weights, data, device='cuda'):
        with torch.no_grad():
            model.eval()
            feat = F.normalize(model.encode(data.to(device)), dim=1)
            proto_norm = F.normalize(fc_weights.to(device), dim=1)
            return (feat @ proto_norm.T).max(dim=1)[0] * self.temperature


# ============================================================
# Pipeline
# ============================================================
def evaluate_session(model, prototypes, test_loader, osr_detector, 
                     num_known, device, args):
    """
    Evaluate on test set: OSR -> classification.
    """
    model.eval()
    all_gt, all_pred = [], []
    all_is_unknown_gt, all_is_unknown_pred = [], []
    known_scores_list, unknown_scores_list = [], []
    proto = prototypes.to(device)
    
    for data, labels in tqdm(test_loader, desc='  Eval'):
        data = data.to(device)
        labels_np = labels.numpy()
        
        with torch.no_grad():
            feat = F.normalize(model.encode(data), dim=1)
            proto_n = F.normalize(proto, dim=1)
            cos_sim = feat @ proto_n.T * args.get('temperature', 10.0)
        
        # OSR detection
        if osr_detector is not None:
            scores = osr_detector.get_scores(model, proto, data, device)
            if isinstance(osr_detector, EnergyOSR) or isinstance(osr_detector, ProtoOSR):
                known_mask = scores > osr_detector.threshold
            else:  # Mahalanobis: lower is more known
                known_mask = scores < osr_detector.threshold
        else:
            known_mask = torch.ones(len(data), dtype=torch.bool, device=device)
        
        is_unk_gt = labels_np >= num_known
        is_unk_pred = ~known_mask.cpu().numpy()
        
        pred_classes = cos_sim.argmax(dim=1).cpu().numpy()
        pred_classes[is_unk_pred] = -1
        
        all_gt.append(labels_np)
        all_pred.append(pred_classes)
        all_is_unknown_gt.append(is_unk_gt)
        all_is_unknown_pred.append(is_unk_pred)
        
        max_sim = cos_sim.max(dim=1)[0].cpu().numpy()
        for i, unk in enumerate(is_unk_gt):
            if unk:
                unknown_scores_list.append(-max_sim[i])
            else:
                known_scores_list.append(-max_sim[i])
    
    return {
        'gt': np.concatenate(all_gt),
        'pred': np.concatenate(all_pred),
        'is_unknown_gt': np.concatenate(all_is_unknown_gt),
        'is_unknown_pred': np.concatenate(all_is_unknown_pred),
        'known_scores': known_scores_list,
        'unknown_scores': unknown_scores_list,
    }


def run_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    dataset_name = args['dataset']
    num_base = args['num_base']
    way = args.get('way', 5)
    num_sessions = args.get('num_novel', 20) // way
    osr_method = args.get('osr_method', 'proto')
    print(f"OSR Method: {osr_method}, Dataset: {dataset_name}, Sessions: {num_sessions}")
    
    # ============ Load Pretrained Model ============
    print("\n=== Loading Pretrained Model ===")
    mynet_args = make_mynet_args(args)
    model = MYNET(mynet_args, mode='encoder')
    pretrained_path = args.get('pretrained_model', '')
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        if 'params' in ckpt:
            state = ckpt['params']
        else:
            state = ckpt
        if any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained model from {pretrained_path}")
    else:
        print(f"WARNING: Pretrained model not found at {pretrained_path}, using random init!")
    model = model.to(device)
    model.mode = 'eval'
    model.eval()
    
    # Base prototypes from fc.weight
    prototypes = model.fc.weight.data[:num_base].clone().detach()
    print(f"Base prototypes: {prototypes.shape}")
    
    # ============ OSR Calibration (fast) ============
    print("\n=== OSR Calibration ===")
    train_ds, val_ds = get_base_loaders(args)
    
    calib_result = calibrate_fast(model, prototypes, val_ds, num_base, osr_method, args, device)
    if osr_method == 'mahalanobis':
        osr_threshold, mahala_means, mahala_invcov = calib_result
    else:
        osr_threshold = calib_result
    print(f"OSR threshold: {osr_threshold}")
    
    # OSR score functions
    T = args.get('temperature', 10.0)
    if osr_method == 'proto':
        def osr_score(feat):
            feat_n = F.normalize(feat, dim=1)
            proto_n = F.normalize(prototypes.to(device), dim=1)
            return (feat_n @ proto_n.T).max(dim=1)[0] * T
        def is_known(feat):
            return osr_score(feat) > osr_threshold
    elif osr_method == 'energy':
        Te = args.get('temperature', 1.0)
        def osr_score(feat):
            logits = F.linear(feat, prototypes.to(device)) / Te
            return -Te * torch.logsumexp(logits / Te, dim=1)
        def is_known(feat):
            return osr_score(feat) > osr_threshold
    elif osr_method == 'mahalanobis':
        _means = mahala_means.to(device)
        _inv = mahala_invcov.to(device)
        def osr_score(feat):
            diff = feat.unsqueeze(1) - _means.unsqueeze(0)
            dists = (diff @ _inv * diff).sum(dim=2)
            return dists.min(dim=1)[0]
        def is_known(feat):
            return osr_score(feat) < osr_threshold
    else:
        def osr_score(feat):
            feat_n = F.normalize(feat, dim=1)
            proto_n = F.normalize(prototypes.to(device), dim=1)
            return (feat_n @ proto_n.T).max(dim=1)[0] * T
        def is_known(feat):
            return osr_score(feat) > osr_threshold
    
    # ============ Incremental Sessions ============
    print("\n=== Incremental Sessions ===")
    all_results = []
    current_protos = prototypes.clone()
    num_known = num_base
    
    for session in range(1, num_sessions + 1):
        print(f"\n--- Session {session}/{num_sessions} ---")
        
        test_loader = get_session_test_loader(args, session)
        
        # Evaluate
        model.eval()
        all_gt, all_pred = [], []
        all_unk_gt, all_unk_pred = [], []
        known_sc, unknown_sc = [], []
        proto = current_protos.to(device)
        
        for data, labels in test_loader:
            data = data.to(device)
            labels_np = labels.numpy()
            with torch.no_grad():
                feat = model.encode(data)
            
            known_mask = is_known(feat)
            is_unk_gt = labels_np >= num_known
            is_unk_pred = ~known_mask.cpu().numpy()
            
            feat_n = F.normalize(feat, dim=1)
            proto_n = F.normalize(proto, dim=1)
            cos_sim = feat_n @ proto_n.T * T
            pred_cls = cos_sim.argmax(dim=1).cpu().numpy()
            pred_cls[is_unk_pred] = -1
            
            all_gt.append(labels_np)
            all_pred.append(pred_cls)
            all_unk_gt.append(is_unk_gt)
            all_unk_pred.append(is_unk_pred)
            
            max_sim = cos_sim.max(dim=1)[0].cpu().numpy()
            for i, unk in enumerate(is_unk_gt):
                (unknown_sc if unk else known_sc).append(-max_sim[i])
        
        all_results.append({
            'gt': np.concatenate(all_gt),
            'pred': np.concatenate(all_pred),
            'is_unknown_gt': np.concatenate(all_unk_gt),
            'is_unknown_pred': np.concatenate(all_unk_pred),
            'known_scores': known_sc,
            'unknown_scores': unknown_sc,
        })
        
        # Get new class prototypes
        new_classes, train_ds = get_new_class_train_set(args, session)
        n_shot = args.get('n_shot', 5)
        all_feats = []
        all_labels_list = []
        
        for cls_idx, cls_id in enumerate(new_classes):
            cls_mask = np.array(train_ds.targets) == cls_id
            cls_indices = np.where(cls_mask)[0]
            chosen = np.random.choice(cls_indices, min(n_shot, len(cls_indices)), replace=False)
            for idx in chosen:
                data, _ = train_ds[idx]
                if isinstance(data, torch.Tensor):
                    if data.dim() == 1:
                        data = data.unsqueeze(0)
                else:
                    data = torch.tensor(data).unsqueeze(0)
                with torch.no_grad():
                    f = model.encode(data.to(device))
                all_feats.append(f.cpu())
                all_labels_list.append(num_known + cls_idx)
        
        all_feats = torch.cat(all_feats, dim=0)
        all_lbls = torch.tensor(all_labels_list)
        
        new_protos = []
        for c in range(num_known, num_known + len(new_classes)):
            mask = all_lbls == c
            if mask.sum() > 0:
                new_protos.append(all_feats[mask].mean(dim=0))
            else:
                new_protos.append(torch.zeros(512))
        
        current_protos = torch.cat([current_protos, torch.stack(new_protos)], dim=0)
        num_known += len(new_classes)
        print(f"  New classes: {new_classes}, Total known: {num_known}")
    
    # ============ Metrics ============
    print("\n=== Computing Metrics ===")
    session_classes = [np.arange(0, num_base + (s+1) * way) for s in range(num_sessions)]
    
    metrics = compute_per_sample_metrics(
        [r['gt'] for r in all_results],
        [r['pred'] for r in all_results],
        [r['is_unknown_gt'] for r in all_results],
        [r['is_unknown_pred'] for r in all_results],
        (0, num_base), session_classes
    )
    
    all_ks = []
    all_us = []
    for r in all_results:
        all_ks.extend(r['known_scores'])
        all_us.extend(r['unknown_scores'])
    auroc, fpr95 = compute_auroc_fpr95(all_ks, all_us, np.zeros(len(all_ks)), np.ones(len(all_us)))
    metrics['AUROC'] = float(auroc)
    metrics['FPR95'] = float(fpr95)
    
    print(f"\nResults for {osr_method} on {dataset_name}:")
    print(f"  AA_inc:      {metrics['AA_inc']:.4f}")
    print(f"  AA_all:      {metrics['AA_all']:.4f}")
    print(f"  AA_known:    {metrics['AA_known']:.4f}")
    print(f"  AA_unknown:  {metrics['AA_unknown']:.4f}")
    print(f"  PD_inc:      {metrics['PD_inc']:.4f}")
    print(f"  AUROC:       {metrics['AUROC']:.4f}")
    print(f"  FPR95:       {metrics['FPR95']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='FOWAC Baseline Runner')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--osr_method', type=str, default='proto',
                        choices=['energy', 'mahalanobis', 'proto'],
                        help='OSR detection method')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='baselines/tables',
                        help='Output directory')
    parser.add_argument('--method_name', type=str, default=None, help='Method name for results')
    args_cli = parser.parse_args()
    
    with open(args_cli.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['osr_method'] = args_cli.osr_method
    config['dataset'] = args_cli.dataset
    
    method_name = args_cli.method_name or args_cli.osr_method
    metrics = run_pipeline(config)
    
    os.makedirs(args_cli.output_dir, exist_ok=True)
    result_path = os.path.join(args_cli.output_dir, f'{method_name}_{args_cli.dataset}.json')
    with open(result_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    main()
