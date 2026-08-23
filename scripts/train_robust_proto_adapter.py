#!/usr/bin/env python3
"""Train a label-free-at-test robust prototype adapter on base validation geometry."""
import argparse
import os
import sys
import random

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.robust_proto_adapter import RobustPrototypeAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=3420)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    geometry = torch.load(args.geometry, map_location='cpu', weights_only=True)
    pools = [x.float() for x in geometry['class_features']]
    centers = geometry['centers'].float().to(device)
    model = RobustPrototypeAdapter(dim=centers.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_classes = len(pools)
    for step in range(args.steps):
        episodes, targets = [], []
        for _ in range(args.batch_size):
            cls = random.randrange(n_classes)
            indices = torch.randperm(len(pools[cls]))[:5]
            support = pools[cls][indices].clone()
            # Match observed CANA noise: usually a 4+1 cluster; retain clean
            # episodes as well so the adapter is identity-safe.
            if random.random() < 0.75:
                other = random.choice([c for c in range(n_classes) if c != cls])
                support[-1] = pools[other][random.randrange(len(pools[other]))]
            episodes.append(support)
            targets.append(centers[cls])
        support = torch.stack(episodes).to(device)
        target = torch.stack(targets).to(device)
        prediction = model(support)
        loss = (1 - F.cosine_similarity(prediction, target, dim=1)).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (step + 1) % 1000 == 0:
            print(f'step={step + 1} loss={loss.item():.6f}')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'dim': centers.shape[1],
                'training': {'base_only': True, 'noise': 'one-of-five',
                             'encoder': 'frozen model.encode', 'offline': True}}, args.output)
    print(f'saved={args.output}')


if __name__ == '__main__':
    main()
