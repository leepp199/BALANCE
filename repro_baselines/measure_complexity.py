"""Measure model complexity (MACs, Parameters, Inference Time).

Usage:
    python -m repro_baselines.measure_complexity --method pitel_cusc

Measures:
- MACs (Multiply-Accumulate operations) using thop
- Number of parameters (in millions)
- Average inference time (AIT) over 100 runs
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Add project root
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from repro_baselines.methods.cil import build_cil, CIL_REGISTRY
from repro_baselines.methods.osr import build_osr, OSR_REGISTRY


def count_encode_macs(model, input_size=(1, 3, 128, 128)):
    """Count MACs of encoder inference path via thop.

    Per MACs measurement protocol: measure only the encode path,
    reflecting real deployment cost.
    """
    from thop import profile
    device = next(model.parameters()).device
    dummy = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    return macs, params


def count_cil_head_macs(feat_dim: int, n_known: int) -> float:
    """CIL head MACs = cosine logit (normalize + matmul) for 1 sample."""
    # normalize feat: O(feat_dim)  (L2 norm)
    # normalize proto: O(n_known * feat_dim)  (L2 norm for each proto)
    # matmul: 2 * feat_dim * n_known  (multiply+accumulate)
    return float(2 * feat_dim * n_known)  # only matmul counted


def count_osr_score_macs(feat_dim: int, n_known: int, method: str) -> float:
    """OSR scoring MACs estimate.

    Most OSR methods compute cosine similarity then some post-processing.
    """
    base = float(2 * feat_dim * n_known)  # similarity matrix
    if method == 'costarr':
        # cos(f, mu_f) + cos(h, mu_h): ~2x base
        return base * 2.0
    if method in ('utl', 'foac_aifp'):
        # known sim + unknown sim: ~2x base
        return base * 2.0
    if method == 'oafn':
        # sim + topk sort: ~base + small
        return base * 1.1
    if method == 'tane':
        # sim + logsumexp: ~base + energy
        return base * 1.5
    # mls, energy: just similarity
    return base


def measure_ait(encoder, input_size=(1, 3, 128, 128),
                n_warmup=50, n_measure=100):
    """Measure average inference time (AIT) of encoder in ms."""
    device = next(encoder.parameters()).device
    dummy = torch.randn(input_size).to(device)
    encoder.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = encoder(dummy)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            start = time.perf_counter()
            _ = encoder(dummy)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times)), float(np.std(times))


def measure_complexity(args):
    """Measure complexity for specified method."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build encoder-only model for complexity measurement
    from models.resnet18_encoder import resnet18
    encoder = resnet18(pretrained=True).to(device).eval()
    # We wrap encode() to mimick MYNET's avgpool + flatten
    class EncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, x):
            x = self.enc(x)
            x = x.mean(dim=[2, 3])  # avgpool
            return x
    model = EncoderWrapper(encoder)

    input_size = (1, 3, 128, 128)
    results = {}

    if args.method in CIL_REGISTRY:
        # Build minimal mock model for CIL method
        mock = nn.Module()
        mock.num_features = args.feat_dim
        mock.fc = nn.Linear(args.feat_dim, args.num_base, bias=False)
        mock.encoder = encoder

        cil_method = build_cil(args.method, mock, args).to(device).eval()

        # 1) Encoder MACs/Params (encode path only)
        enc_macs, enc_params = count_encode_macs(model, input_size)
        results['Encoder_MACs'] = enc_macs
        results['Encoder_Params'] = enc_params

        # 2) CIL head MACs (computed analytically)
        n_known = args.num_base
        head_macs = count_cil_head_macs(args.feat_dim, n_known)
        results['CIL_Head_MACs'] = head_macs
        # CIL head params = n_known * feat_dim (prototypes)
        results['CIL_Head_Params'] = float(n_known * args.feat_dim)

        results['Total_MACs'] = results['Encoder_MACs'] + results['CIL_Head_MACs']
        results['Total_Params'] = results['Encoder_Params'] + results['CIL_Head_Params']

        # 3) AIT
        ait, ait_std = measure_ait(model, input_size)
        results['AIT_ms'] = ait
        results['AIT_std_ms'] = ait_std

    elif args.method in OSR_REGISTRY:
        osr_method = build_osr(args.method, args)

        # OSR score MACs (estimated analytically)
        score_macs = count_osr_score_macs(
            args.feat_dim, args.num_base, args.method)
        results['OSR_Score_MACs'] = score_macs

        osr_p = 0
        for attr_name in dir(osr_method):
            attr = getattr(osr_method, attr_name)
            if isinstance(attr, nn.Module):
                for p in attr.parameters():
                    if p.requires_grad:
                        osr_p += p.numel()
        if hasattr(osr_method, 'parameters'):
            for p in osr_method.parameters():
                if p.requires_grad:
                    osr_p += p.numel()
        results['OSR_Learned_Params'] = osr_p
        results['Total_MACs'] = results.get('OSR_Score_MACs', 0)
        results['Total_Params'] = results.get('OSR_Learned_Params', 0)

    # Print & save
    print(f"\n{'='*50}")
    print(f"Complexity: {args.method}")
    print(f"{'='*50}")
    for k, v in results.items():
        if 'MACs' in k:
            print(f"  {k}: {v:.2e}")
        elif 'Params' in k:
            print(f"  {k}: {v/1e6:.4f}M")
        elif 'AIT' in k:
            print(f"  {k}: {v:.3f}")

    out_dir = osp.join(ROOT, 'repro_baselines', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f'complexity_{args.method}.txt')
    with open(out_path, 'w') as fp:
        for k, v in results.items():
            fp.write(f"{k}: {v}\n")
    print(f"\nSaved to: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        help='Method name (CIL or OSR)')
    parser.add_argument('--num_base', type=int, default=80)
    parser.add_argument('--num_all', type=int, default=100)
    parser.add_argument('--feat_dim', type=int, default=512)
    args = parser.parse_args()

    measure_complexity(args)


if __name__ == '__main__':
    main()
