#!/usr/bin/env python3
"""Offline base-only episodic adaptation for the current FSC encoder."""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataloader import get_pretrain_dataloader, seed_worker
from data.sampler import CategoriesSampler
from network import MYNET
from train_unopenset import args_parser, dict2namespace, set_seed, set_up_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/exp_fsc89.yml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=16.0)
    parser.add_argument('--layers', default='layer4')
    parser.add_argument('--augment', action='store_true')
    cli = parser.parse_args()

    with open(cli.config) as handle:
        file_cfg = yaml.safe_load(handle)['train']
    cfg = vars(args_parser().parse_args([]))
    cfg.update(file_cfg)
    cfg.update({'dataset': 'FMC',
                'dataroot': '/data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data'})
    args = dict2namespace(cfg)
    args.dataloader.num_workers = 0
    set_seed(args.seed)
    set_up_datasets(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MYNET(args, mode='incre').to(device)
    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint.get('cls_params', checkpoint.get('params', checkpoint))
    model.load_state_dict(state, strict=False)
    trainset, _ = get_pretrain_dataloader(args)
    sampler = CategoriesSampler(trainset.targets, n_batch=cli.episodes,
                                n_cls=5, n_per=20)
    loader = torch.utils.data.DataLoader(
        trainset, batch_sampler=sampler, num_workers=0, pin_memory=True,
        worker_init_fn=seed_worker)

    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    layer_names = [name.strip() for name in cli.layers.split(',') if name.strip()]
    for name in layer_names:
        layer = getattr(model.encoder, name)
        for parameter in layer.parameters():
            parameter.requires_grad = True
        trainable.extend(layer.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=cli.lr, weight_decay=1e-4)
    target = torch.arange(5, device=device).repeat(15)

    for epoch in range(cli.epochs):
        model.eval()
        for name in layer_names:
            getattr(model.encoder, name).train()
        losses, accuracies = [], []
        for waveforms, _ in loader:
            waveforms = waveforms.to(device)
            features = (model.base_encode(waveforms, augment=True)
                        if cli.augment else model.encode(waveforms))
            support = features[:25].view(5, 5, -1)
            query = features[25:].view(15, 5, -1).reshape(75, -1)
            prototypes = support.mean(0)
            logits = cli.temperature * F.linear(
                F.normalize(query, dim=-1), F.normalize(prototypes, dim=-1))
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            accuracies.append(float((logits.argmax(1) == target).float().mean()))
        print(f'epoch={epoch + 1} loss={sum(losses)/len(losses):.6f} '
              f'query_acc={sum(accuracies)/len(accuracies):.6f}')

    os.makedirs(os.path.dirname(cli.output), exist_ok=True)
    torch.save({'params': model.state_dict(), 'episodic_training': {
        'base_classes': 69, 'novel_classes_seen': False,
        'ways': 5, 'shots': 5, 'queries': 15,
        'episodes': cli.episodes, 'epochs': cli.epochs,
        'lr': cli.lr, 'layers': layer_names, 'augment': cli.augment,
        'offline': True}}, cli.output)
    print(f'saved={cli.output}')


if __name__ == '__main__':
    main()
