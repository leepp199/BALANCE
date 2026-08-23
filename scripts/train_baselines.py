"""Train each CIL baseline independently (base + optional meta).

Produces per-method checkpoints for fair comparison:
    save/baseline_{method}_{dataset}.pth

Usage:
    python -m scripts.train_baselines --config configs/baseline_eval_ls.yml \
        --methods cec amfo pan --gpu 0

    python -m scripts.train_baselines --config configs/baseline_eval_ns.yml \
        --methods cec amfo pan --gpu 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataloader import get_pretrain_dataloader  # noqa: E402
from models.baselines import build_cil  # noqa: E402
from network import MYNET  # noqa: E402
from utils.utils import set_gpu  # noqa: E402


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
    parser.add_argument('--methods', nargs='+', default=['cec', 'amfo', 'pan'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--log_dir', type=str, default='save/train_logs')
    cli = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = cli.gpu
    os.makedirs(cli.save_dir, exist_ok=True)
    os.makedirs(cli.log_dir, exist_ok=True)

    args = load_args(cli.config)
    from train_unopenset import set_up_datasets
    set_up_datasets(args)
    args.cuda = torch.cuda.is_available()

    device = torch.device('cuda' if args.cuda else 'cpu')
    dataset_tag = getattr(args, 'dataset', 'default')

    for method in cli.methods:
        print(f"\n{'='*60}")
        print(f" Training method={method} on dataset={dataset_tag}")
        print(f"{'='*60}\n")
        t0 = time.time()

        # Fresh model for each method
        model = MYNET(args, mode='encoder').to(device)

        # Build CIL wrapper (it grabs model.fc.weight as initial protos)
        cil = build_cil(method, model, args).to(device)

        # ---- Base training ----
        _, trainloader = get_pretrain_dataloader(args)
        log_path = os.path.join(cli.log_dir, f'{method}_{dataset_tag}.log')
        cil.train_base(args, trainloader, log_path=log_path)

        # ---- Save checkpoint ----
        ckpt_path = os.path.join(cli.save_dir, f'baseline_{method}_{dataset_tag}.pth')
        state = {
            'params': model.state_dict(),
            'cil_state': cil.state_dict(),
            'method': method,
            'dataset': dataset_tag,
        }
        torch.save(state, ckpt_path)
        elapsed = time.time() - t0
        print(f"\n>>> {method} done in {elapsed/60:.1f} min. Saved to {ckpt_path}")


if __name__ == '__main__':
    main()
