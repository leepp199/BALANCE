"""Unified driver to evaluate the 9 baseline combinations.

For every (CIL, OSR) pair we run the full session-incremental pipeline:

    session 0  : pure base-class evaluation (no OSR)
    session s≥1: features -> OSR scoring -> unknown/known split
                 -> k-means on unknowns -> CIL prototype registration
                 -> all / inc / known / unknown accuracy

Results are written to ``save_result/baselines/{cil}_{osr}.txt`` in the
same text format as ``train_unopenset``; a combined summary CSV is
written to ``save_result/baselines/comparison_table.csv``.

Usage (from repo root)::

    python -m scripts.run_all_baselines --config configs/default.yml \
        --pretrained save/base_train_for_meta_ls.pth \
        --cil cec amfo pan --osr mls tane nci
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

# project root on path -------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataloader import (  # noqa: E402
    get_dataloader,
    get_inc_testloader,
    get_pretrain_dataloader,
    get_testloader,
)
from models.baselines import build_cil, build_osr  # noqa: E402
from network import MYNET, replace_base_fc  # noqa: E402
from utils.util import cluster_acc, calc  # noqa: E402
from utils.utils import set_gpu, count_acc, Averager  # noqa: E402


# ======================================================================
# Config loader (minimal YAML parser reusing the project convention)
# ======================================================================

def load_args(config_path: str, cli_overrides: dict = None) -> argparse.Namespace:
    """Load YAML config and flatten like train_unopenset does.
    
    Matches the merge logic in train_unopenset.py:
        cfg = yaml.safe_load(f)['train']
        cfg.update(vars(args))
        args = dict2namespace(cfg)
    """
    import yaml

    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    # Extract the 'train' section (same as train_unopenset) and flatten
    cfg = raw.get('train', raw)
    if cli_overrides:
        cfg.update(cli_overrides)

    def dict2namespace(d):
        ns = argparse.Namespace()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(ns, k, dict2namespace(v))
            else:
                setattr(ns, k, v)
        return ns

    return dict2namespace(cfg)


# ======================================================================
# Utility — encode a loader into feature tensor
# ======================================================================

@torch.no_grad()
def encode_loader(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labs = [], []
    model.eval()
    prev_mode = model.mode
    model.mode = 'incre'
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        f = model.encode(x)
        feats.append(f.cpu())
        labs.append(y.cpu())
    model.mode = prev_mode
    if not feats:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(feats, 0), torch.cat(labs, 0)


# ======================================================================
# Core per-session baseline routine
# ======================================================================

def evaluate_session(model, args, session, cil, osr, device) -> Dict[str, float]:
    # 1. features of the mixed (known+unknown) session-s loader
    _, mixed_loader = get_dataloader(args, session)
    feats, labs = encode_loader(model, mixed_loader, device)
    feats = feats.to(device)

    # 2. OSR scoring → higher = more unknown
    n_known = args.num_labeled_classes
    proto_bank = model.fc.weight[:n_known, :].detach().to(device)
    scores = osr.score(feats, proto_bank)
    # AUROC: binary classification (known=0, unknown=1)
    y_true = (labs >= n_known).cpu().numpy()
    y_score = scores.cpu().numpy()
    # Ensure both classes present to avoid roc_auc_score error
    if len(np.unique(y_true)) >= 2:
        auroc = float(roc_auc_score(y_true, y_score))
    else:
        auroc = 0.0

    thr = torch.quantile(scores, 0.5)
    unk_mask = (scores >= thr).cpu()
    unk_feats = feats[unk_mask.to(feats.device)]
    unk_labs = labs[unk_mask]
    kn_feats = feats[(~unk_mask).to(feats.device)]
    kn_labs = labs[~unk_mask]

    # 3. cluster unknowns → pseudo labels → register novel protos via CIL
    n_clusters = args.num_unlabeled_classes
    cluster_a = 0.0
    novel_proto_bank = None
    if unk_feats.shape[0] >= n_clusters:
        km = KMeans(n_clusters=n_clusters, n_init=20).fit(unk_feats.cpu().numpy())
        cluster_a, mapping = cluster_acc(args, unk_labs.numpy(), km.labels_)
        novel_ids = []
        novel_protos = []
        for c in range(n_clusters):
            tgt = mapping.get(c, None)
            if tgt is None or tgt < args.num_labeled_classes:
                continue
            idx = torch.from_numpy(km.labels_ == c)
            if idx.sum() == 0:
                continue
            proto = unk_feats[idx.to(unk_feats.device)].mean(0)
            novel_ids.append(int(tgt))
            novel_protos.append(proto)
        if novel_protos:
            novel_proto_bank = torch.stack(novel_protos, 0)
            # Build per-sample support features + pseudo labels for PAN (needs them)
            km_labels_t = torch.from_numpy(km.labels_).long()
            nids_t = torch.tensor(novel_ids, dtype=torch.long)
            # Map cluster ids → novel class ids for per-sample features
            km_to_cid = {}
            for c in range(n_clusters):
                tgt = mapping.get(c, None)
                if tgt is not None and tgt >= args.num_labeled_classes:
                    km_to_cid[c] = int(tgt)
            # Gather features belonging to novel clusters
            nov_mask = torch.tensor([km.labels_[i] in km_to_cid for i in range(unk_feats.shape[0])])
            nov_feats = unk_feats[nov_mask.to(unk_feats.device)]
            nov_pseudo = torch.tensor([km_to_cid[km.labels_[i]]
                                       for i in range(unk_feats.shape[0])
                                       if km.labels_[i] in km_to_cid], dtype=torch.long)
            # Try PAN-style call (feats + labels), fall back to proto-only for CEC/AMFO
            try:
                cil.register_novel_classes(nov_feats, novel_ids, labels=nov_pseudo)
            except TypeError:
                cil.register_novel_classes(novel_proto_bank, novel_ids)
            # push CIL prototypes into model.fc so test() sees them
            with torch.no_grad():
                proto_all = cil._protos.data.to(model.fc.weight.device)
                need = max(proto_all.size(0), model.fc.weight.size(0))
                if need > model.fc.weight.size(0):
                    pad = torch.zeros(need - model.fc.weight.size(0),
                                       model.fc.weight.size(1),
                                       device=model.fc.weight.device)
                    model.fc.weight.data = torch.cat([model.fc.weight.data, pad], 0)
                model.fc.weight.data[:proto_all.size(0)] = proto_all

    # 4. known accuracy on the split
    known_acc = 0.0
    if kn_feats.numel() > 0:
        proto_now = model.fc.weight[:args.num_labeled_classes, :].detach()
        logits = F.cosine_similarity(kn_feats.unsqueeze(1), proto_now, dim=-1)
        known_acc = count_acc(logits, kn_labs.to(device))

    # 5. advance class pool and evaluate all / inc
    args.num_labeled_classes += args.way
    _, all_loader = get_testloader(args, session)
    all_acc = _plain_cosine_eval(model, all_loader, args.num_labeled_classes, device)
    _, inc_loader = get_inc_testloader(args, session)
    inc_acc = _plain_cosine_eval(model, inc_loader, args.num_labeled_classes, device)

    fscore = calc(args, kn_labs.tolist(), unk_labs.tolist())

    return dict(known=float(known_acc), unknown=float(cluster_a),
                auroc=auroc, f1=float(fscore), inc=float(inc_acc), all=float(all_acc))


@torch.no_grad()
def _plain_cosine_eval(model, loader, n_class, device) -> float:
    model.eval()
    prev_mode = model.mode
    model.mode = 'incre'
    ave = Averager()
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        f = model.encode(x)
        proto = model.fc.weight[:n_class, :].detach()
        logits = F.cosine_similarity(f.unsqueeze(1), proto, dim=-1)
        ave.add(count_acc(logits, y))
    model.mode = prev_mode
    return float(ave.item())


# ======================================================================
# Driver loop
# ======================================================================

def run_combo(args, base_state, cil_name: str, osr_name: str,
              out_dir: str, cil_state: dict = None,
              test_times: int = 1) -> Dict[str, List[float]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tag = f"{cil_name}_{osr_name}"
    path = os.path.join(out_dir, f"{tag}.txt")

    # Multi-run aggregation
    all_records = []
    for trial in range(test_times):
        model = MYNET(args, mode='encoder').to(device)
        model.load_state_dict(base_state, strict=False)
        if trial > 0:
            # Reset seed per trial for reproducibility
            torch.manual_seed(trial)
            np.random.seed(trial)

        # align classifier prototypes with the pretrained features
        trainset, _ = get_pretrain_dataloader(args)
        model = replace_base_fc(args, trainset, model)

        cil = build_cil(cil_name, model, args).to(device)
        if cil_state is not None:
            cil.load_state_dict(cil_state, strict=False)
        osr = build_osr(osr_name, args)

        # Fit OSR methods that require labelled training features (e.g. FOAC_AIFP)
        if hasattr(osr, 'fit'):
            train_loader = get_pretrain_dataloader(args)[1]
            osr.fit(model, train_loader)

        args.num_labeled_classes = args.num_base

        session0 = _plain_cosine_eval(model, get_testloader(args, 0)[1],
                                       args.num_base, device)
        record = {k: [] for k in ['known', 'unknown', 'auroc', 'f1', 'inc', 'all']}
        for s in range(args.start_session, args.num_session):
            m = evaluate_session(model, args, s, cil, osr, device)
            for k in record:
                record[k].append(m[k])

        record['session0'] = session0
        all_records.append(record)

    # Aggregate over trials: compute mean across trials
    n_sessions = args.num_session - args.start_session
    record_agg = {k: [] for k in ['known', 'unknown', 'auroc', 'f1', 'inc', 'all']}
    for s in range(n_sessions):
        for k in record_agg:
            vals = [r[k][s] for r in all_records]
            record_agg[k].append(float(np.mean(vals)))
    session0_mean = float(np.mean([r['session0'] for r in all_records]))
    record_agg['session0'] = session0_mean

    with open(path, 'w') as fp:
        fp.write(f"=== Baseline: CIL={cil_name} OSR={osr_name} ===\n")
        fp.write(f"Session 0 acc={session0_mean:.4f} (avg over {test_times} runs)\n")
        fp.write(f"test_times={test_times}\n")
        for s in range(n_sessions):
            fp.write("session:{},known:{:.4f},unknown:{:.4f},auroc:{:.4f},f1:{:.4f},inc:{:.4f},all:{:.4f}\n".format(
                s + 1,
                record_agg['known'][s], record_agg['unknown'][s],
                record_agg['auroc'][s], record_agg['f1'][s],
                record_agg['inc'][s], record_agg['all'][s]))

    return record_agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yml')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='single base-trained checkpoint (.pth) — used for all CIL methods')
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='directory with per-method checkpoints: baseline_{cil}_{ds}.pth')
    parser.add_argument('--cil', nargs='+', default=['cec', 'amfo', 'pan'])
    parser.add_argument('--osr', nargs='+', default=['mls', 'tane', 'nci'])
    parser.add_argument('--out_dir', type=str,
                        default='save_result/baselines')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset name (e.g. librispeech, nsynth-100, FMC)')
    parser.add_argument('--dataroot', type=str, default=None,
                        help='dataset root path')
    parser.add_argument('--test_times', type=int, default=10,
                        help='number of evaluation repeats (default: 10)')
    cli = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = cli.gpu
    os.makedirs(cli.out_dir, exist_ok=True)

    # Build CLI overrides for config flattening
    cli_overrides = {}
    if cli.dataset:
        cli_overrides['dataset'] = cli.dataset
    if cli.dataroot:
        cli_overrides['dataroot'] = cli.dataroot

    args = load_args(cli.config, cli_overrides or None)
    # set_up_datasets fills args.Dataset / args.num_base / args.num_session …
    from train_unopenset import set_up_datasets
    set_up_datasets(args)
    args.cuda = torch.cuda.is_available()
    args.start_session = getattr(args, 'start_session', 1)

    # Resolve checkpoint: either single --pretrained or per-method --pretrained_dir
    dataset_tag = getattr(args, 'dataset', 'default')

    def _get_base_state(cil_name: str) -> tuple:
        """Return (base_state, cil_state) dict for the given CIL method."""
        if cli.pretrained_dir:
            # Per-method independent checkpoint
            ckpt_path = os.path.join(cli.pretrained_dir,
                                     f'baseline_{cil_name}_{dataset_tag}.pth')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Per-method checkpoint not found: {ckpt_path}\n"
                    f"Run scripts/train_baselines.py first.")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            return ckpt.get('params', ckpt), ckpt.get('cil_state', None)
        elif cli.pretrained:
            ckpt = torch.load(cli.pretrained, map_location='cpu')
            return ckpt.get('params', ckpt), None
        else:
            raise ValueError("Must specify --pretrained or --pretrained_dir")

    # CSV header
    csv_path = os.path.join(cli.out_dir, 'comparison_table.csv')
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['cil', 'osr', 'session0', 'AA_all', 'AA_inc', 'PD_all',
                         *[f's{i}_all' for i in range(1, args.num_session)],
                         *[f's{i}_inc' for i in range(1, args.num_session)]])

        for c in cli.cil:
            base_state, cil_state = _get_base_state(c)
            for o in cli.osr:
                t0 = time.time()
                rec = run_combo(args, base_state, c, o, cli.out_dir,
                                cil_state=cil_state, test_times=cli.test_times)
                aa_all = float(np.mean(rec['all']))
                aa_inc = float(np.mean(rec['inc']))
                pd_all = float(rec['session0'] - rec['all'][-1])
                writer.writerow([c, o,
                                 round(rec['session0'], 4),
                                 round(aa_all, 4), round(aa_inc, 4),
                                 round(pd_all, 4),
                                 *[round(x, 4) for x in rec['all']],
                                 *[round(x, 4) for x in rec['inc']]])
                print(f"=== {c}×{o} done in {time.time()-t0:.1f}s "
                      f"AA_all={aa_all:.4f} AA_inc={aa_inc:.4f} PD={pd_all:.4f} ===")

    print(f"\nComparison table saved to {csv_path}")


if __name__ == '__main__':
    main()
