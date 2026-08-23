#!/usr/bin/env python3
"""Fit an offline residual map from current model.encode centers to fc weights."""
import argparse
import os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--ridge', type=float, required=True)
    a = p.parse_args()
    geometry = torch.load(a.geometry, map_location='cpu', weights_only=True)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint.get('params', checkpoint.get('cls_params', checkpoint))
    x = geometry['centers'].float()
    y = state['fc.weight'][:len(x)].float()
    d = x.shape[1]
    eye = torch.eye(d)
    # Identity-centred ridge: min ||XW-Y||^2 + lambda ||W-I||^2.
    weight = torch.linalg.solve(x.t() @ x + a.ridge * eye,
                                x.t() @ y + a.ridge * eye)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    torch.save({'weight': weight, 'ridge': a.ridge,
                'training': {'offline': True, 'encoder': 'current model.encode',
                             'target': 'trained fc.weight', 'test_classes_seen': False}}, a.output)
    before = torch.nn.functional.cosine_similarity(x, y, dim=1).mean()
    after = torch.nn.functional.cosine_similarity(x @ weight, y, dim=1).mean()
    print(f'ridge={a.ridge:g} cosine_before={before:.6f} cosine_after={after:.6f} output={a.output}')


if __name__ == '__main__':
    main()
