"""Task 3.4 — uncertainty / curriculum training curves.

Parses ``extract_log.txt`` (or any training log passed via --log) for
keys such as ``center_loss``, ``hard_weight``, ``mc_dropout_var``, and
plots them against epoch.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt


_PATTERNS = {
    'center_loss':      re.compile(r'center[_ ]loss[^\d\-]*([\d\.\-eE]+)', re.I),
    'cls_loss':         re.compile(r'cls[_ ]loss[^\d\-]*([\d\.\-eE]+)',    re.I),
    'hard_weight':      re.compile(r'hard[_ ]weight[^\d\-]*([\d\.\-eE]+)', re.I),
    'mc_var':           re.compile(r'(?:mc[_ ]?var|mc[_ ]?dropout)[^\d\-]*([\d\.\-eE]+)', re.I),
    'curriculum_ratio': re.compile(r'curriculum[_ ]ratio[^\d\-]*([\d\.\-eE]+)', re.I),
    'train_acc':        re.compile(r'train[_ ]acc[^\d\-]*([\d\.\-eE]+)',   re.I),
}


def parse_log(path: str):
    series = {k: [] for k in _PATTERNS}
    epochs = []
    cur_epoch = -1
    with open(path, 'r', errors='ignore') as fp:
        for line in fp:
            m = re.search(r'epoch\s*[:=]?\s*(\d+)', line, re.I)
            if m:
                cur_epoch = int(m.group(1))
                if not epochs or epochs[-1] != cur_epoch:
                    epochs.append(cur_epoch)
                    for k in series:
                        series[k].append(None)
            for k, pat in _PATTERNS.items():
                m = pat.search(line)
                if m and series[k]:
                    try:
                        series[k][-1] = float(m.group(1))
                    except ValueError:
                        pass
    return epochs, series


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', default='extract_log.txt')
    p.add_argument('--out_dir', default='save/figures/curves')
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if not os.path.exists(a.log):
        print(f'log not found: {a.log}')
        return
    epochs, series = parse_log(a.log)
    if not epochs:
        print('no epoch info parsed')
        return

    keys = [k for k, v in series.items() if any(x is not None for x in v)]
    ncols = min(3, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows),
                              squeeze=False)
    for i, k in enumerate(keys):
        r, c = i // ncols, i % ncols
        ax = axes[r][c]
        xs = [e for e, v in zip(epochs, series[k]) if v is not None]
        ys = [v for v in series[k] if v is not None]
        ax.plot(xs, ys, marker='.')
        ax.set_title(k); ax.set_xlabel('epoch'); ax.grid(alpha=0.3)
    for j in range(len(keys), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.tight_layout()
    out = os.path.join(a.out_dir, 'curriculum_curves.png')
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
