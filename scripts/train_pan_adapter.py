#!/usr/bin/env python3
"""Train APGM/PQAM on pseudo-incremental base episodes, fully offline."""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataloader import get_base_dataloader_stdu
from network import MYNET
from train_unopenset import args_parser, dict2namespace, set_seed, set_up_datasets
from utils.utils import count_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/exp_fsc89.yml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--tmp-base', type=int, default=44)
    p.add_argument('--tmp-novel', type=int, default=20)
    cli = p.parse_args()
    if not os.path.isfile(cli.checkpoint):
        raise FileNotFoundError(f'Offline checkpoint missing: {cli.checkpoint}')

    with open(cli.config) as handle:
        file_cfg = yaml.safe_load(handle)['train']
    cfg = vars(args_parser().parse_args([]))
    cfg.update(file_cfg)
    cfg.update({'dataset': 'FMC',
                'dataroot': '/data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data'})
    args = dict2namespace(cfg)
    args.stdu.num_tmpb = cli.tmp_base
    args.stdu.num_tmpi = cli.tmp_novel
    args.stdu.pqa = True
    args.episode.train_episode = cli.episodes
    args.dataloader.num_workers = 0
    set_seed(args.seed)
    set_up_datasets(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MYNET(args, mode='incre').to(device)
    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint.get('cls_params', checkpoint.get('params', checkpoint))
    model.load_state_dict(state, strict=False)
    _, loader = get_base_dataloader_stdu(args)

    optimizer = torch.optim.SGD([
        {'params': model.encoder.parameters(), 'lr': 2e-4},
        {'params': model.slf_attn.parameters(), 'lr': 2e-4},
        {'params': model.transatt_proto.parameters(), 'lr': 2e-4},
    ], momentum=0.9, nesterov=True, weight_decay=args.optimizer.decay)

    low_way, novel_way = args.episode.low_way, args.episode.episode_way
    low_shot, novel_shot = args.episode.low_shot, args.episode.episode_shot
    query_count = args.episode.episode_query
    target = torch.arange(low_way + novel_way, device=device).repeat(query_count)
    model.eval()  # match the paper code: gradients enabled, BN statistics frozen
    for epoch in range(cli.epochs):
        losses, accs = [], []
        for batch, _ in loader:
            batch = batch.to(device)
            base_total = low_way * (low_shot + query_count)
            base_support_end = low_way * low_shot
            novel_support_end = base_total + novel_way * novel_shot

            base = model.encode(batch[:base_total])
            novel_support = model.encode(batch[base_total:novel_support_end])
            novel_query = model.encode(batch[novel_support_end:])
            base_support, base_query = base[:base_support_end], base[base_support_end:]
            support_embeddings = torch.cat([base_support, novel_support], dim=0)

            base_proto = base_support.view(low_shot, low_way, -1).mean(0, keepdim=True)
            novel_proto = novel_support.view(novel_shot, novel_way, -1).mean(0, keepdim=True)
            proto = torch.cat([base_proto, novel_proto], dim=1).unsqueeze(0)
            query = torch.cat([
                base_query.view(query_count, low_way, -1),
                novel_query.view(query_count, novel_way, -1),
            ], dim=1).unsqueeze(0)
            logits, _, _ = model._forward(
                proto, query, pqa=True, sup_emb=support_embeddings,
                novel_ids=torch.arange(low_way + novel_way, device=device))
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            accs.append(float(count_acc(logits.detach(), target)))
        print(f'epoch={epoch + 1} loss={sum(losses)/len(losses):.6f} '
              f'acc={sum(accs)/len(accs):.6f}')

    os.makedirs(os.path.dirname(cli.output), exist_ok=True)
    torch.save({'params': model.state_dict(), 'pan_training': {
        'epochs': cli.epochs, 'episodes': cli.episodes,
        'tmp_base': cli.tmp_base, 'tmp_novel': cli.tmp_novel,
        'offline': True}}, cli.output)
    print(f'saved={cli.output}')


if __name__ == '__main__':
    main()
