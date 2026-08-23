#!/usr/bin/env python3
"""Paired stream-level significance for one complete FOWAC method."""
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(
    r'\[RAW\] round=(?P<round>\d+) session=(?P<session>[1-4]).*?'
    r'acc known=(?P<known>[0-9.]+).*?acc unknown=(?P<unknown>[0-9.]+).*?'
    r'auroc=(?P<auroc>[0-9.]+).*?f1=(?P<f1>[0-9.]+).*?'
    r'inc=(?P<inc>[0-9.]+).*?all=(?P<all>[0-9.]+)')

def rounds(path):
    rows = [m.groupdict() for m in LINE.finditer((ROOT / path).read_text(errors='replace'))]
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col])
    complete = frame.groupby('round').filter(lambda x: len(x) == 4)
    return complete.groupby('round')[['known', 'unknown', 'auroc', 'f1', 'inc', 'all']].mean()

pairs = [
    ('LS-100', 'logs/ls100_rank25_late_frozen_50.log', 'logs/raw_v2_teen_ls100_10runs.log'),
    ('NS-100', 'logs/ns_all_q50_cana_50final.log', 'logs/raw_v2_teen_ns100_10runs.log'),
]
rng = np.random.default_rng(3420)
records = []
for dataset, proposed_path, baseline_path in pairs:
    proposed, baseline = rounds(proposed_path), rounds(baseline_path)
    common = proposed.index.intersection(baseline.index)
    for metric in proposed.columns:
        delta = proposed.loc[common, metric].to_numpy() - baseline.loc[common, metric].to_numpy()
        if not len(delta):
            continue
        samples = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(1)
        records.append({
            'method': 'FOWAC', 'baseline': 'TEEN', 'dataset': dataset,
            'metric': metric, 'n_pairs': len(delta), 'mean_gain': delta.mean(),
            'gain_ci95_low': np.quantile(samples, .025),
            'gain_ci95_high': np.quantile(samples, .975),
            'paired_t_p': stats.ttest_1samp(delta, 0).pvalue if len(delta) > 1 else np.nan,
            'wilcoxon_p': stats.wilcoxon(delta).pvalue if len(delta) > 1 and np.any(delta) else np.nan,
        })
result = pd.DataFrame(records)
result.to_csv(ROOT / 'experiments/fowac_paired_significance.csv', index=False)

show = result[result.metric.isin(['inc', 'all', 'auroc', 'f1'])].copy()
show['label'] = show.dataset + ' / ' + show.metric.str.upper()
show = show.sort_values('mean_gain')
y = np.arange(len(show)); gain = show.mean_gain.to_numpy() * 100
lo = show.gain_ci95_low.to_numpy() * 100; hi = show.gain_ci95_high.to_numpy() * 100
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'pdf.fonttype': 42, 'axes.linewidth': .7})
fig, ax = plt.subplots(figsize=(5.2, max(2.2, .32*len(show)+.8)))
ax.errorbar(gain, y, xerr=[gain-lo, hi-gain], fmt='o', ms=4,
            color='#0072B2', ecolor='#6B7280', capsize=2, lw=.9)
ax.axvline(0, color='#B91C1C', ls='--', lw=.8)
ax.set_yticks(y, show.label)
ax.set_xlabel('FOWAC − TEEN paired gain (percentage points; 95% bootstrap CI)')
ax.grid(axis='x', color='#D1D5DB', lw=.5, alpha=.75)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(ROOT / 'figures' / f'fowac_paired_significance.{ext}', dpi=400,
                bbox_inches='tight')
print(f'wrote {len(result)} paired tests')
