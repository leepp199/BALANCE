"""Task 3.5 — 9-method main-table line/bar chart.

Reads ``save_result/baselines/comparison_table.csv`` (produced by
``scripts/run_all_baselines``) and renders:

* a per-session line plot of all-class accuracy for every (CIL, OSR) pair
* a grouped bar chart for AA_all, AA_inc, PD_all
* optionally overlays the "Ours" row if ``--ours_csv`` is provided
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def read_table(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='save_result/baselines/comparison_table.csv')
    p.add_argument('--ours_csv', default=None,
                   help='optional row csv with the same columns for our method')
    p.add_argument('--out_dir', default='save/figures/main_table')
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = read_table(a.csv)
    if a.ours_csv and os.path.exists(a.ours_csv):
        rows.extend(read_table(a.ours_csv))

    sess_cols = [k for k in rows[0].keys() if k.startswith('s') and k.endswith('_all')]
    inc_cols = [k for k in rows[0].keys() if k.startswith('s') and k.endswith('_inc')]

    # -------- line plot: per-session all-acc --------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sess_idx = list(range(1, len(sess_cols) + 1))
    for r in rows:
        label = f"{r['cil'].upper()}-{r['osr'].upper()}"
        ys = [float(r[k]) for k in sess_cols]
        ax.plot(sess_idx, ys, marker='o', label=label)
    ax.set_xlabel('session'); ax.set_ylabel('all-class acc')
    ax.set_title('Per-session all-class accuracy')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'main_lineplot.png'), dpi=160)
    plt.close(fig)

    # -------- grouped bar: AA_all / AA_inc / PD --------
    keys = ['AA_all', 'AA_inc', 'PD_all']
    labels = [f"{r['cil'].upper()}-{r['osr'].upper()}" for r in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels)), 4.5))
    for i, k in enumerate(keys):
        vals = [float(r[k]) for r in rows]
        ax.bar(x + (i - 1) * width, vals, width, label=k)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('value'); ax.legend()
    ax.set_title('Aggregate metrics across methods')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'main_bar.png'), dpi=160)
    plt.close(fig)

    print(f'saved figures to {a.out_dir}')


if __name__ == '__main__':
    main()
