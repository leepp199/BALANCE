#!/usr/bin/env python3
"""Paired significance for metric-complete BCD versus TEEN on FSC-89."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


source = Path('experiments/extended_metrics_sessions.csv')
output = Path('experiments/paired_extended_significance.csv')
if not source.exists():
    raise SystemExit('run collect_extended_metrics.py first')

frame = pd.read_csv(source)
base = frame[(frame.method == 'TEEN') & (frame.dataset == 'FSC-89')]
prop = frame[(frame.method == 'FOWAC-BCD') & (frame.dataset == 'FSC-89')]
rows = []
rng = np.random.default_rng(3420)
for metric in ('auroc', 'aupr', 'fpr95', 'cluster_acc', 'nmi', 'ari'):
    b = base.groupby('repeat')[metric].mean().sort_index().to_numpy()
    p = prop.groupby('repeat')[metric].mean().sort_index().to_numpy()
    n = min(len(b), len(p))
    b, p = b[:n], p[:n]
    diff = p - b
    if n:
        boot = np.mean(rng.choice(diff, size=(20000, n), replace=True), axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
    else:
        lo = hi = np.nan
    t_p = float(ttest_rel(p, b).pvalue) if n >= 2 and np.std(diff) > 0 else np.nan
    try:
        w_p = float(wilcoxon(diff).pvalue) if n >= 2 and np.any(diff) else np.nan
    except ValueError:
        w_p = np.nan
    rows.append({
        'method': 'FOWAC-BCD', 'dataset': 'FSC-89', 'metric': metric,
        'n_pairs': n, 'baseline_mean': b.mean() if n else np.nan,
        'proposed_mean': p.mean() if n else np.nan,
        'mean_gain': diff.mean() if n else np.nan, 'gain_ci95_low': lo,
        'gain_ci95_high': hi, 'paired_t_p': t_p, 'wilcoxon_p': w_p,
        'complete_10_pairs': n == 10,
    })

with output.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(f'wrote {len(rows)} rows to {output}')
