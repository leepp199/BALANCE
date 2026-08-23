#!/usr/bin/env python3
"""Collect AUPR/FPR95 and NMI/ARI from metric-complete repeated logs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


OSR = re.compile(
    r'\[OSR-METRIC\] session=(?P<session>[1-4]) AUROC=(?P<auroc>[0-9.]+) '
    r'AUPR=(?P<aupr>[0-9.]+) FPR95=(?P<fpr95>[0-9.]+)'
)
CLU = re.compile(
    r'\[CLU-METRIC\] session=(?P<session>[1-4]) ACC=(?P<acc>[0-9.]+) '
    r'NMI=(?P<nmi>[0-9.]+) ARI=(?P<ari>-?[0-9.]+)'
)
ROUND = re.compile(r'=== Base Session Pure Evaluation \(Round \d+\) ===')

SPECS = [
    ('TEEN', 'LS-100', 'logs/raw_v2_teen_ls100_10runs.log'),
    ('TEEN', 'NS-100', 'logs/raw_v2_teen_ns100_10runs.log'),
    ('TEEN', 'FSC-89', 'logs/raw_v2_teen_fsc89_10runs.log'),
    ('FOWAC-UMR', 'LS-100', 'logs/raw_v2_umr_ls100_10runs.log'),
    ('FOWAC-DS', 'LS-100', 'logs/raw_v2_dfsb_ds_ls100_10runs.log'),
    ('FOWAC-BCD', 'FSC-89', 'logs/raw_v2_bcd_fsc89_10runs.log'),
]


def complete_rounds(text):
    """Align metrics by explicit repeat boundary; zero-fill skipped discovery."""
    result = []
    for segment in ROUND.split(text)[1:]:
        osr = {int(m.group('session')): m.groupdict() for m in OSR.finditer(segment)}
        clu = {int(m.group('session')): m.groupdict() for m in CLU.finditer(segment)}
        if sorted(osr) != [1, 2, 3, 4]:
            continue
        result.append((osr, clu))
    return result


raw = []
summary = []
for method, dataset, source in SPECS:
    path = Path(source)
    if not path.exists():
        continue
    text = path.read_text(errors='replace')
    rounds = complete_rounds(text)
    n = len(rounds)
    if not n:
        continue
    for repeat in range(n):
        osr_by_session, clu_by_session = rounds[repeat]
        for session in range(1, 5):
            o = osr_by_session[session]
            c = clu_by_session.get(session)
            raw.append({
                'method': method, 'dataset': dataset, 'repeat': repeat,
                'session': session, 'auroc': float(o['auroc']),
                'aupr': float(o['aupr']), 'fpr95': float(o['fpr95']),
                'cluster_acc': float(c['acc']) if c else 0.0,
                'nmi': float(c['nmi']) if c else 0.0,
                'ari': float(c['ari']) if c else 0.0, 'source_log': source,
            })
    own = [row for row in raw if row['method'] == method and row['dataset'] == dataset]
    record = {'method': method, 'dataset': dataset, 'n_repeats': n, 'source_log': source}
    for key in ('auroc', 'aupr', 'fpr95', 'cluster_acc', 'nmi', 'ari'):
        repeat_means = np.asarray([
            np.mean([r[key] for r in own if r['repeat'] == repeat]) for repeat in range(n)
        ])
        record[f'{key}_mean'] = repeat_means.mean()
        record[f'{key}_std'] = repeat_means.std(ddof=1) if n > 1 else 0.0
    summary.append(record)

raw_path = Path('experiments/extended_metrics_sessions.csv')
summary_path = Path('experiments/extended_metrics_summary.csv')
if raw:
    with raw_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
if summary:
    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
print(f'wrote {len(raw)} session rows and {len(summary)} summary rows')
