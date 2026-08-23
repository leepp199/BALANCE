#!/usr/bin/env python
"""Export the common acoustic feature stream consumed by the official VB-CGCD code.

This deliberately keeps feature extraction in the project's PyTorch 3.8 environment and
writes plain NumPy archives, so the official Python 3.12/JAX environment never changes the
audio model or sample ordering.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_unopenset import dict2namespace, set_seed, set_up_datasets
from network import MYNET
from data.dataloader import (get_know_dataloader, get_mixed_openworld_dataloader,
                             get_testloader, get_inc_testloader)


def encode_loader(model, loader, device):
    xs, ys = [], []
    model.eval()
    model.mode = 'incre'
    with torch.no_grad():
        for audio, label in loader:
            feature = model.encode(audio.to(device))
            if feature.ndim > 2:
                feature = feature.flatten(1)
            xs.append(feature.detach().cpu().float().numpy())
            ys.append(label.detach().cpu().long().numpy())
    return np.concatenate(xs), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=3420)
    args_cli = parser.parse_args()

    with open(args_cli.config) as handle:
        cfg = yaml.safe_load(handle)['train']
    cfg.update(dataset=args_cli.dataset, dataroot=args_cli.dataroot, seed=args_cli.seed)
    cfg.setdefault('train_weight_base', 1)
    cfg.setdefault('base_seman_calib', 1)
    cfg.setdefault('neg_gen_type', 'att')
    cfg.setdefault('agg', 'avg')
    cfg.setdefault('num_labeled_classes', cfg['num_base'])
    cfg.setdefault('num_unlabeled_classes', cfg['way'])
    args = dict2namespace(cfg)
    set_seed(args.seed)
    set_up_datasets(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MYNET(args, mode='encoder').to(device)
    checkpoint = torch.load(args_cli.checkpoint, map_location=device)
    params = checkpoint['cls_params'] if 'cls_params' in checkpoint else checkpoint['params']
    if 'cls_params' in checkpoint:
        cls_params = {key: value for key, value in params.items() if 'fc' in key}
        model.cls_classifier.init_representation(cls_params)
    state = model.state_dict()
    state.update(params)
    model.load_state_dict(state)

    output = Path(args_cli.output)
    output.mkdir(parents=True, exist_ok=True)

    _, base_loader = get_know_dataloader(args, session=0)
    train_x, train_y = encode_loader(model, base_loader, device)
    _, base_test_loader = get_testloader(args, session=0)
    test_x, test_y = encode_loader(model, base_test_loader, device)
    np.savez_compressed(output / 'session_0.npz', train_x=train_x, train_y=train_y,
                        test_all_x=test_x, test_all_y=test_y,
                        test_old_x=test_x, test_old_y=test_y,
                        test_novel_x=test_x, test_novel_y=test_y)

    for session in range(1, args.num_session):
        args.current_test = 0
        _, stream_loader = get_mixed_openworld_dataloader(args, session)
        stream_x, stream_y = encode_loader(model, stream_loader, device)
        _, all_loader = get_testloader(args, session)
        all_x, all_y = encode_loader(model, all_loader, device)
        old_mask = all_y < args.num_base + (session - 1) * args.way
        _, novel_loader = get_inc_testloader(args, session)
        novel_x, novel_y = encode_loader(model, novel_loader, device)
        np.savez_compressed(
            output / f'session_{session}.npz',
            train_x=stream_x, train_y=stream_y,
            test_all_x=all_x, test_all_y=all_y,
            test_old_x=all_x[old_mask], test_old_y=all_y[old_mask],
            test_novel_x=novel_x, test_novel_y=novel_y,
        )
        print(f'session={session} stream={len(stream_y)} all_test={len(all_y)}')


if __name__ == '__main__':
    main()
