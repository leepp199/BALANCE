"""Run FEC-OSL end-to-end baseline evaluation.

Evaluates the full open-world pipeline using FEC-OSL's energy-based
detection + adaptive clustering, then reports per-session metrics.

Usage:
    python -m scripts.run_fec_osl --config configs/baseline_eval_ls.yml \
        --dataroot /data/datasets/librispeech_fscil/ --gpu 0

Results written to ``{out_dir}/fec_osl_{dataset}.txt``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataloader import (
    get_dataloader,
    get_inc_testloader,
    get_pretrain_dataloader,
    get_testloader,
)
from models.baselines.end_to_end.fec_osl import FEC_OSL


def load_args(config_path: str, cli_overrides: dict = None):
    import yaml
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--out_dir', type=str, default='save_result/end_to_end')
    parser.add_argument('--test_times', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    cli = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = cli.gpu
    os.makedirs(cli.out_dir, exist_ok=True)

    cli_overrides = {}
    if cli.dataset:
        cli_overrides['dataset'] = cli.dataset
    if cli.dataroot:
        cli_overrides['dataroot'] = cli.dataroot

    args = load_args(cli.config, cli_overrides or None)
    from train_unopenset import set_up_datasets
    set_up_datasets(args)
    args.cuda = torch.cuda.is_available()
    args.start_session = getattr(args, 'start_session', 1)

    dataset_tag = getattr(args, 'dataset', 'default')

    print(f"\n{'='*60}")
    print(f" FEC-OSL evaluation on dataset={dataset_tag}")
    print(f"{'='*60}")

    # Multi-run evaluation
    all_runs = []
    for trial in range(cli.test_times):
        torch.manual_seed(trial)
        np.random.seed(trial)

        learner = FEC_OSL(args)
        learner.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Base training
        _, trainloader = get_pretrain_dataloader(args)
        learner.train_base(trainloader, epochs=cli.epochs)

        # Session 0
        session0 = float(learner._eval_all(
            get_testloader(args, 0)[1], args.num_base))

        records = {k: [] for k in ['known', 'unknown', 'auroc', 'f1', 'inc', 'all']}
        known_classes = args.num_base
        for s in range(args.start_session, args.num_session):
            _, mixed_loader = get_dataloader(args, s)
            m = learner.evaluate_session(mixed_loader, s, known_classes)
            for k in records:
                records[k].append(m[k])
            known_classes += args.way

        all_runs.append({'session0': session0, **records})

    # Aggregate
    n_sessions = args.num_session - args.start_session
    agg = {k: [] for k in ['known', 'unknown', 'auroc', 'f1', 'inc', 'all']}
    for s in range(n_sessions):
        for k in agg:
            vals = [r[k][s] for r in all_runs]
            agg[k].append(float(np.mean(vals)))
    s0_mean = float(np.mean([r['session0'] for r in all_runs]))

    # Write results
    path = os.path.join(cli.out_dir, f'fec_osl_{dataset_tag}.txt')
    with open(path, 'w') as fp:
        fp.write(f"=== FEC-OSL on {dataset_tag} ===\n")
        fp.write(f"Session 0 acc={s0_mean:.4f} (avg over {cli.test_times} runs)\n")
        fp.write(f"test_times={cli.test_times}\n")
        for s in range(n_sessions):
            fp.write(f"session:{s+1},known:{agg['known'][s]:.4f},"
                     f"unknown:{agg['unknown'][s]:.4f},auroc:{agg['auroc'][s]:.4f},f1:{agg['f1'][s]:.4f},"
                     f"inc:{agg['inc'][s]:.4f},all:{agg['all'][s]:.4f}\n")

    aa_all = float(np.mean(agg['all']))
    aa_inc = float(np.mean(agg['inc']))
    pd_all = float(s0_mean - agg['all'][-1])

    # CSV summary
    csv_path = os.path.join(cli.out_dir, 'comparison.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as cf:
        writer = csv.writer(cf)
        if write_header:
            writer.writerow(['method', 'dataset', 'session0', 'AA_all', 'AA_inc',
                             'PD_all'] +
                            [f's{i}_all' for i in range(1, args.num_session)] +
                            [f's{i}_inc' for i in range(1, args.num_session)])
        writer.writerow(['fec_osl', dataset_tag,
                         round(s0_mean, 4), round(aa_all, 4), round(aa_inc, 4),
                         round(pd_all, 4),
                         *[round(agg['all'][i], 4) for i in range(n_sessions)],
                         *[round(agg['inc'][i], 4) for i in range(n_sessions)]])

    print(f"\nResults: s0={s0_mean:.4f} AA_all={aa_all:.4f} AA_inc={aa_inc:.4f} PD={pd_all:.4f}")
    print(f"Saved to {path}")

    # Also compute summary
    print(f"\n=== FEC-OSL {dataset_tag} Summary ===")
    for s in range(n_sessions):
        print(f"  S{s+1}: all={agg['all'][s]:.4f} inc={agg['inc'][s]:.4f}")
    print(f"  AA_all={aa_all:.4f}  AA_inc={aa_inc:.4f}  PD_all={pd_all:.4f}")


if __name__ == '__main__':
    main()
