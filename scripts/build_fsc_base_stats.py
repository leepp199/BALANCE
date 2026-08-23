#!/usr/bin/env python3
"""Build offline base-class geometry from model.encode features."""
import argparse
import os
import sys

import numpy as np
import scipy.linalg
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from network import MYNET
from train_unopenset import args_parser, dict2namespace, set_seed, set_up_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/exp_fsc89.yml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', default='FMC', choices=['FMC', 'librispeech', 'nsynth-100'])
    parser.add_argument('--dataroot', default='/data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data')
    parser.add_argument('--phase', default='val', choices=['train', 'val'])
    cli = parser.parse_args()
    with open(cli.config) as handle:
        file_cfg = yaml.safe_load(handle)['train']
    cfg = vars(args_parser().parse_args([]))
    cfg.update(file_cfg)
    cfg.update({'dataset': cli.dataset, 'dataroot': cli.dataroot})
    args = dict2namespace(cfg)
    args.dataloader.num_workers = 0
    set_seed(args.seed)
    set_up_datasets(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MYNET(args, mode='incre').to(device).eval()
    state = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('params', state), strict=False)
    dataset_cls = getattr(args.Dataset, {
        'FMC': 'FSDCLIPS', 'librispeech': 'LBRS', 'nsynth-100': 'NDS'
    }[cli.dataset])
    dataset = dataset_cls(root=args.dataroot, phase=cli.phase,
                          index=np.arange(args.num_base), base_sess=True, args=args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False,
                                         num_workers=0, pin_memory=True)
    per_class = [[] for _ in range(args.num_base)]
    with torch.no_grad():
        for waves, labels in loader:
            model.mode = 'extract_feature'
            features = model.encode(waves.to(device)).cpu()
            for feature, label in zip(features, labels):
                per_class[int(label)].append(feature)
    centers, radius_mean, radius_std = [], [], []
    for features in per_class:
        x = torch.stack(features)
        center = x.mean(0)
        distances = torch.linalg.vector_norm(x - center, dim=1)
        centers.append(center)
        radius_mean.append(distances.mean())
        radius_std.append(distances.std().clamp_min(1e-6))
    centers_tensor = torch.stack(centers)
    global_center = centers_tensor.mean(0)
    sw = torch.zeros(centers_tensor.shape[1], centers_tensor.shape[1])
    for features, center in zip(per_class, centers):
        centered = torch.stack(features) - center
        sw += centered.t().mm(centered)
    centered_means = centers_tensor - global_center
    sb = centered_means.t().mm(centered_means)
    sw_np = (sw / max(len(dataset) - len(centers), 1)).numpy()
    sb_np = sb.numpy()
    regularizer = 1e-3 * float(np.trace(sw_np) / sw_np.shape[0])
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        sb_np, sw_np + regularizer * np.eye(sw_np.shape[0]))
    order = np.argsort(eigenvalues)[::-1]
    lda_dim = min(len(centers) - 1, 128)
    lda_projection = torch.from_numpy(eigenvectors[:, order[:lda_dim]].astype(np.float32))
    os.makedirs(os.path.dirname(cli.output), exist_ok=True)
    torch.save({'centers': centers_tensor,
                'radius_mean': torch.stack(radius_mean),
                'radius_std': torch.stack(radius_std),
                'lda_projection': lda_projection,
                'class_features': [torch.stack(features) for features in per_class],
                'split': f'{cli.dataset} {cli.phase}', 'encoder': 'model.encode',
                'offline': True}, cli.output)
    print(f'saved={cli.output} classes={len(centers)} samples={len(dataset)}')


if __name__ == '__main__':
    main()
