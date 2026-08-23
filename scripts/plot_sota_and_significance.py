#!/usr/bin/env python3
"""Compact paper figures for compatible SOTA results and paired statistics."""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

datasets = ['LS-100', 'NS-100', 'FSC-89']
methods = ['TEEN', 'OFCL', 'OPCR', 'YLOC', 'Happy', 'FOWAC']
labels = {
    'OFCL-acoustic-transfer': 'OFCL', 'OPCR-acoustic': 'OPCR',
    'YLOC-PAL-acoustic': 'YLOC', 'Happy-HAProto-acoustic': 'Happy'
}
metrics = [('inc_aa', 'Average incremental accuracy'),
           ('all_aa', 'Average all-class accuracy'),
           ('auroc_aa', 'OSR AUROC'), ('osr_f1_aa', 'OSR F1')]

# Protocol-compatible visual-method transfers (ten algorithm seeds).
transfer = pd.read_csv(ROOT / 'experiments/feature_transfer_three_datasets.csv')
transfer['method'] = transfer.method.map(labels)
agg = transfer.groupby(['method', 'dataset']).agg(
    inc_aa=('inc_aa', 'mean'), all_aa=('all_aa', 'mean'),
    auroc_aa=('auroc_aa', 'mean'), osr_f1_aa=('osr_f1_aa', 'mean')).reset_index()

# Canonical TEEN repeated streams.
raw = pd.read_csv(ROOT / 'experiments/raw_v2_summary.csv')
teen = raw[raw.method.eq('TEEN')].pivot(index='dataset', columns='metric', values='aa_mean')
for dataset in datasets:
    if dataset in teen.index:
        row = teen.loc[dataset]
        agg.loc[len(agg)] = ['TEEN', dataset, row.get('inc'), row.get('all'),
                             row.get('auroc'), row.get('f1')]

# Current audited FOWAC rows. These are replaced by the unified 50-repeat logs
# before submission; keeping the source explicit prevents accidental mixing.
raw_line = re.compile(
    r'\[RAW\].*?acc known=(?P<known>[0-9.]+).*?acc unknown=(?P<unknown>[0-9.]+)'
    r'.*?auroc=(?P<auroc>[0-9.]+).*?f1=(?P<f1>[0-9.]+).*?inc=(?P<inc>[0-9.]+)'
    r'.*?all=(?P<all>[0-9.]+)')
def audited(path):
    rows = [m.groupdict() for m in raw_line.finditer((ROOT / path).read_text(errors='replace'))]
    frame = pd.DataFrame(rows, dtype=float)
    return tuple(frame[key].mean() for key in ('inc', 'all', 'auroc', 'f1'))

current = {
    'LS-100': audited('logs/ls100_rank25_late_frozen_50.log'),
    'NS-100': audited('logs/ns_all_q50_cana_50final.log'),
}
for dataset, values in current.items():
    agg.loc[len(agg)] = ['FOWAC', dataset, *values]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2,
                     'pdf.fonttype': 42, 'axes.linewidth': .7})
fig = plt.figure(figsize=(7.15, 4.45))
grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, .82], wspace=.34, hspace=.44)
axes = np.asarray([[fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
                   [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]])
status_ax = fig.add_subplot(grid[:, 2])
colors = {'LS-100': '#0072B2', 'NS-100': '#D55E00', 'FSC-89': '#009E73'}
x = np.arange(len(methods))
for ax, (metric, title) in zip(axes.flat, metrics):
    for offset, dataset in zip((-.18, 0, .18), datasets):
        vals = []
        for method in methods:
            hit = agg[(agg.method == method) & (agg.dataset == dataset)]
            vals.append(float(hit.iloc[0][metric]) * 100 if len(hit) else np.nan)
        ax.scatter(x + offset, vals, s=20, color=colors[dataset], label=dataset,
                   edgecolor='white', linewidth=.35, zorder=3)
    ax.set_title(title, weight='semibold')
    ax.set_xticks(x, methods, rotation=23, ha='right')
    ax.set_ylabel('(%)')
    ax.grid(axis='y', color='#D1D5DB', lw=.5, alpha=.75)
    ax.spines[['top', 'right']].set_visible(False)
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, ncol=3, loc='upper center', frameon=False,
           bbox_to_anchor=(.5, .995))
status = [
    ('TEEN', 'Transfer'), ('OFCL', 'Transfer'), ('OPCR', 'Reimpl.'),
    ('YLOC', 'Reimpl.'), ('Happy', 'Reimpl.'), ('FOWAC', 'Acoustic'),
    ('CAMP', 'Visual only'), ('MetaGCD', 'Visual only'),
    ('OCGCD/DEAN', 'Visual only'), ('OpenIncrement', 'Visual only'),
    ('VB-CGCD', 'DNF'), ('FaE', 'Visual only'), ('PRISM', 'Visual only'),
    ('OCCD', 'Visual only'), ('VC-CGCD', 'Visual only')]
status_names = [x[0] for x in status]
status_types = ['Acoustic', 'Transfer', 'Reimpl.', 'Visual only', 'DNF']
status_colors = {'Acoustic': '#0072B2', 'Transfer': '#56B4E9',
                 'Reimpl.': '#E69F00', 'Visual only': '#9CA3AF', 'DNF': '#B91C1C'}
for yi, (_, kind) in enumerate(status):
    xi = status_types.index(kind)
    status_ax.scatter(xi, yi, s=25, color=status_colors[kind], zorder=3)
status_ax.set_xticks(range(len(status_types)), status_types, rotation=45, ha='right', fontsize=6)
status_ax.set_yticks(range(len(status_names)), status_names, fontsize=6.2)
status_ax.invert_yaxis()
status_ax.set_title('Protocol / provenance', weight='semibold', fontsize=8)
status_ax.grid(color='#E5E7EB', lw=.45)
status_ax.spines[['top', 'right']].set_visible(False)
fig.text(.5, .005, 'Visual-only rows are shown for coverage, not numerically mixed with acoustic results.',
         ha='center', fontsize=6.2)
fig.subplots_adjust(left=.075, right=.99, top=.90, bottom=.19)
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'sota_compact_comparison.{ext}', dpi=400, bbox_inches='tight')
plt.close(fig)

# Forest plot of paired effects already backed by per-stream pairs.
sig = pd.read_csv(ROOT / 'experiments/fowac_paired_significance.csv')
sig = sig[sig.metric.isin(['inc', 'all', 'auroc', 'f1'])].copy()
sig['label'] = sig['method'] + ' / ' + sig['dataset'] + ' / ' + sig['metric'].str.upper()
sig = sig.sort_values('mean_gain')
y = np.arange(len(sig))
fig, ax = plt.subplots(figsize=(7.15, max(2.5, .25 * len(sig) + .7)))
gain = sig.mean_gain.to_numpy() * 100
lo = sig.gain_ci95_low.to_numpy() * 100
hi = sig.gain_ci95_high.to_numpy() * 100
ax.errorbar(gain, y, xerr=[gain-lo, hi-gain], fmt='o', ms=3.5,
            color='#0072B2', ecolor='#6B7280', capsize=2, lw=.8)
ax.axvline(0, color='#B91C1C', ls='--', lw=.8)
ax.set_yticks(y, sig.label)
ax.set_xlabel('Paired gain over TEEN (percentage points; 95% bootstrap CI)')
ax.grid(axis='x', color='#D1D5DB', lw=.5, alpha=.75)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'paired_significance_forest.{ext}', dpi=400, bbox_inches='tight')
print('wrote SOTA comparison and paired-significance figures')
