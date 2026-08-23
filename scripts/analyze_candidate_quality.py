#!/usr/bin/env python
"""Quantify how discovery-buffer coverage/purity predicts incremental accuracy."""
import csv
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
specs = {
    'LS-100/TEEN': 'logs/sota_three_datasets/teen_pure_ls100_canonical_10runs.log',
    'LS-100/FOWAC': 'logs/proposed_statmem_cachedvar_ls100_10runs.log',
    'NS-100/TEEN': 'logs/sota_three_datasets/teen_pure_ns100_10runs.log',
    'FSC-89/TEEN': 'logs/sota_three_datasets/teen_pure_fsc89_10runs.log',
}
candidate_re = re.compile(r'Input to KMeans: total=(\d+) true_unknown=(\d+) true_known_leaked=(\d+)')
session_re = re.compile(r'session:([1-4]),[^\n]*?incremental acc:([0-9.]+)')
rows = []
for key, rel in specs.items():
    dataset, method = key.split('/')
    pending = []
    repeat = 0
    for line in (ROOT / rel).read_text(errors='replace').splitlines():
        match = candidate_re.search(line)
        if match:
            pending.append(tuple(map(int, match.groups())))
            continue
        match = session_re.search(line)
        if match and pending:
            session, inc = int(match.group(1)), float(match.group(2))
            total, unknown, leaked = pending.pop(0)
            if session == 1:
                repeat += 1
            rows.append({'dataset': dataset, 'method': method, 'repeat': repeat,
                         'session': session, 'candidate_total': total,
                         'novel_coverage': unknown / 25.0,
                         'candidate_purity': unknown / max(total, 1),
                         'known_leakage': leaked / max(total, 1),
                         'incremental_acc': inc})

out = ROOT / 'experiments' / 'candidate_quality_sessions.csv'
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

summary = []
for dataset in sorted({r['dataset'] for r in rows}):
    group = [r for r in rows if r['dataset'] == dataset and r['method'] == 'TEEN']
    y = np.asarray([r['incremental_acc'] for r in group])
    for metric in ('novel_coverage', 'candidate_purity', 'known_leakage'):
        x = np.asarray([r[metric] for r in group])
        pr, pp = pearsonr(x, y) if np.std(x) and np.std(y) else (np.nan, np.nan)
        sr, sp = spearmanr(x, y) if np.std(x) and np.std(y) else (np.nan, np.nan)
        summary.append({'dataset': dataset, 'metric': metric, 'n': len(group),
                        'pearson_r': pr, 'pearson_p': pp,
                        'spearman_rho': sr, 'spearman_p': sp})
summary_out = ROOT / 'experiments' / 'candidate_quality_correlation.csv'
with summary_out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)
print(f'wrote {len(rows)} session rows and {len(summary)} correlations')
