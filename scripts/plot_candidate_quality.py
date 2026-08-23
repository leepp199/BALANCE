#!/usr/bin/env python
"""Plot the dominant discovery-buffer bottleneck for each dataset."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / 'experiments' / 'candidate_quality_sessions.csv')
data = data[data.method == 'TEEN']
choices = [('LS-100', 'candidate_purity', 'Candidate purity'),
           ('NS-100', 'novel_coverage', 'Novel-class coverage'),
           ('FSC-89', 'candidate_purity', 'Candidate purity')]
colors = ['#0072B2', '#009E73', '#D55E00']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.7, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
for ax, (dataset, metric, xlabel), color in zip(axes, choices, colors):
    group = data[data.dataset == dataset]
    x, y = group[metric].to_numpy() * 100, group.incremental_acc.to_numpy() * 100
    ax.scatter(x, y, s=17, color=color, alpha=0.65, edgecolor='white', linewidth=0.35)
    coefficient = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, np.polyval(coefficient, grid), color=color, lw=1.5)
    r, p = pearsonr(x, y)
    ax.text(0.04, 0.95, f'$r={r:.2f}$\n$p={p:.2g}$', transform=ax.transAxes,
            va='top', ha='left', fontsize=7.5,
            bbox={'facecolor': 'white', 'edgecolor': '#D1D5DB', 'pad': 2.5})
    ax.set_title(dataset, weight='semibold', pad=4)
    ax.set_xlabel(f'{xlabel} (%)')
    ax.grid(color='#D1D5DB', lw=0.5, alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Incremental accuracy (%)')
fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.86, wspace=0.28)
out = ROOT / 'figures'
for suffix in ('pdf', 'png'):
    fig.savefig(out / f'candidate_quality_correlation.{suffix}', dpi=400, bbox_inches='tight')
print(f'wrote {out / "candidate_quality_correlation.pdf"}')
