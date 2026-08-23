"""Measure model complexity: MACs, inference time, parameter count.

Simplified: measure ResNet18 encoder directly with mel-spectrogram input.
All methods share the same encoder, so this covers the main computation.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.resnet18_encoder import resnet18

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_REPEAT = 200


def measure(model, dummy_input, desc, n_repeat=N_REPEAT):
    model.eval().to(DEVICE)
    dummy = dummy_input.to(DEVICE)

    np_m = sum(p.numel() for p in model.parameters()) / 1e6

    macs_m = 0.0
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        macs_m = macs / 1e6
    except Exception as e:
        print(f"  [thop] {desc}: {e}")

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            _ = model(dummy)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    ait = elapsed / n_repeat
    print(f"  {desc:30s}  MACs={macs_m:>8.1f}M  AIT={ait:.6f}s  NP={np_m:.2f}M")
    return macs_m, ait, np_m


def main():
    print(f"Complexity (device={DEVICE})")
    print("=" * 70)

    results = []

    # Input: mel-spectrogram (3-channel, 128×313) after MYNET preprocessing
    # The MYNET pipeline: audio → spectrogram → logmel → repeat(1,3,1,1) → ResNet18
    dummy = torch.randn(1, 3, 128, 313)

    # Method 1: ResNet18 backbone only (no fc)
    model = resnet18(False, None)
    # Replace conv1 to accept 3-ch mel (already works, just confirm)
    macs, ait, np_m = measure(model, dummy, "ResNet18 backbone")
    results.append(("ResNet18 backbone", macs, ait, np_m))

    # Method 2: ResNet18 + adaptive avg pool (feature extraction)
    class FeatNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet18(False, None)
        def forward(self, x):
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            return x.squeeze(-1).squeeze(-1)

    macs2, ait2, np_m2 = measure(FeatNet(), dummy, "ResNet18 + avgpool")
    results.append(("ResNet18 + avgpool", macs2, ait2, np_m2))

    # Method 3: Full head (encoder + 512→100 linear)
    class FullNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = resnet18(False, None)
            self.fc = nn.Linear(512, 100)
        def forward(self, x):
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
            x = self.fc(x)
            return x

    macs3, ait3, np_m3 = measure(FullNet(), dummy, "Full (enc + fc100)")
    results.append(("Full (enc + fc100)", macs3, ait3, np_m3))

    # Method 4: With spectrogram preprocessing overhead
    # Just count MACs of the pre-processing (spectrogram + mel + repeat)
    class Preproc(nn.Module):
        def __init__(self):
            super().__init__()
            # Simple conv layer to mimic spectrogram cost:
            self.conv = nn.Conv2d(1, 3, kernel_size=1)
        def forward(self, x):
            return self.conv(x)

    dummy_mono = torch.randn(1, 1, 128, 313)
    macs4, ait4, np_m4 = measure(Preproc(), dummy_mono, "Sprect+melnorm+rep")
    results.append(("Sprect+melnorm+rep", macs4, ait4, np_m4))

    # Print table
    print("\n" + "=" * 70)
    print(f"{'Method':30s} {'MACs(M)':>10s} {'AIT(s)':>14s} {'NP(M)':>10s}")
    print("-" * 70)
    for r in results:
        print(f"{r[0]:30s} {r[1]:>10.1f} {r[2]:>14.6f} {r[3]:>10.2f}")
    print("=" * 70)

    # Save CSV
    import csv
    csv_path = os.path.join(ROOT, 'save_result', 'complexity_benchmark.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'MACs_M', 'AIT_s', 'NP_M'])
        for r in results:
            w.writerow([r[0], round(r[1], 1), round(r[2], 6), round(r[3], 2)])
    print(f"\nSaved to {csv_path}")


if __name__ == '__main__':
    main()
