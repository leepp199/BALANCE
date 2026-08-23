#!/usr/bin/env python
"""Paired repeat-level comparison between pure TEEN and the proposed method."""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


LINE = re.compile(
    r'\[RAW\] round=(?P<round>\d+) session=(?P<session>[1-4]) '
    r'acc known=(?P<known>[0-9.]+) acc unknown=(?P<unknown>[0-9.]+) '
    r'auroc=(?P<auroc>[0-9.]+) f1=(?P<f1>[0-9.]+) '
    r'inc=(?P<inc>[0-9.]+) all=(?P<all>[0-9.]+)')


def repeats(path):
    path = Path(path)
    if not path.exists():
        return []
    entries = [m.groupdict() for m in LINE.finditer(path.read_text(errors='replace'))]
    result = []
    by_round = {}
    for entry in entries:
        by_round.setdefault(int(entry['round']), []).append(entry)
    for round_id in sorted(by_round):
        block = by_round[round_id]
        if len(block) != 4 or [int(x['session']) for x in block] != [1, 2, 3, 4]:
            continue
        record = {k: np.mean([float(x[k]) for x in block])
                  for k in ('known', 'unknown', 'auroc', 'f1', 'inc', 'all')}
        record['final_inc'] = float(block[-1]['inc'])
        result.append(record)
    return result


def bootstrap_ci(diff, seed=3420, draws=20000):
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(diff, size=(draws, len(diff)), replace=True), axis=1)
    return np.quantile(means, [0.025, 0.975])


def compare(method, dataset, baseline_path, proposed_path):
    base, prop = repeats(baseline_path), repeats(proposed_path)
    n = min(len(base), len(prop))
    rows = []
    for metric in ('known', 'unknown', 'auroc', 'f1', 'inc', 'final_inc', 'all'):
        b = np.asarray([x[metric] for x in base[:n]])
        p = np.asarray([x[metric] for x in prop[:n]])
        d = p - b
        lo, hi = bootstrap_ci(d) if n else (np.nan, np.nan)
        t_p = float(ttest_rel(p, b).pvalue) if n >= 2 and np.std(d) > 0 else np.nan
        try: w_p = float(wilcoxon(d).pvalue) if n >= 2 and np.any(d) else np.nan
        except ValueError: w_p = np.nan
        rows.append({'method': method, 'dataset': dataset, 'metric': metric, 'n_pairs': n,
                     'baseline_mean': b.mean() if n else np.nan,
                     'proposed_mean': p.mean() if n else np.nan,
                     'mean_gain': d.mean() if n else np.nan,
                     'gain_ci95_low': lo, 'gain_ci95_high': hi,
                     'paired_t_p': t_p, 'wilcoxon_p': w_p,
                     'complete_10_pairs': n == 10})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='experiments/paired_significance.csv')
    a = p.parse_args()
    specs = [
        ('FOWAC-UMR', 'LS-100', 'logs/raw_v2_teen_ls100_10runs.log', 'logs/raw_v2_umr_ls100_10runs.log'),
        ('FOWAC-DS', 'LS-100', 'logs/raw_v2_teen_ls100_10runs.log', 'logs/raw_v2_dfsb_ds_ls100_10runs.log'),
        ('FOWAC-BCD', 'FSC-89', 'logs/raw_v2_teen_fsc89_10runs.log', 'logs/raw_v2_bcd_fsc89_10runs.log'),
    ]
    rows = []
    for spec in specs: rows.extend(compare(*spec))
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} metric rows to {out}')


if __name__ == '__main__':
    main()
