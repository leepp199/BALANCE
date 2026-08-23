#!/usr/bin/env python3
"""Generate traceable current paper tables; never invent missing SOTA cells."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'PAPER_RESULT_TABLES_CURRENT.md'

transfer = pd.read_csv(ROOT / 'experiments/feature_transfer_three_datasets.csv')
rename = {'OFCL-acoustic-transfer': 'OFCL', 'OPCR-acoustic': 'OPCR',
          'YLOC-PAL-acoustic': 'YLOC', 'Happy-HAProto-acoustic': 'Happy'}
transfer['method'] = transfer.method.map(rename)
sota = transfer.groupby(['method', 'dataset', 'provenance'], as_index=False).agg(
    repeats=('seed', 'nunique'), inc_aa=('inc_aa', 'mean'), all_aa=('all_aa', 'mean'),
    auroc_aa=('auroc_aa', 'mean'), osr_f1_aa=('osr_f1_aa', 'mean'))

line = re.compile(
    r'\[RAW\].*?acc known=(?P<known>[0-9.]+).*?acc unknown=(?P<unknown>[0-9.]+)'
    r'.*?auroc=(?P<auroc>[0-9.]+).*?f1=(?P<f1>[0-9.]+).*?inc=(?P<inc>[0-9.]+)'
    r'.*?all=(?P<all>[0-9.]+)')
osr = re.compile(r'\[OSR-METRIC\].*?AUPR=(?P<aupr>[0-9.]+) FPR95=(?P<fpr95>[0-9.]+)')
def log_stats(path):
    text = (ROOT / path).read_text(errors='replace')
    frame = pd.DataFrame([m.groupdict() for m in line.finditer(text)], dtype=float)
    extra = pd.DataFrame([m.groupdict() for m in osr.finditer(text)], dtype=float)
    result = {key: frame[key].mean() for key in frame}
    result.update({key: extra[key].mean() for key in extra} if len(extra) else {})
    result['repeats'] = len(frame) // 4
    return result

logs = {
    ('TEEN', 'LS-100'): 'logs/raw_v2_teen_ls100_10runs.log',
    ('TEEN', 'NS-100'): 'logs/raw_v2_teen_ns100_10runs.log',
    ('TEEN', 'FSC-89'): 'logs/raw_v2_teen_fsc89_10runs.log',
    ('FOWAC', 'LS-100'): 'logs/ls100_rank25_late_frozen_50.log',
    ('FOWAC', 'NS-100'): 'logs/ns_all_q50_cana_50final.log',
}
detail = []
for (method, dataset), path in logs.items():
    row = log_stats(path)
    row.update(method=method, dataset=dataset, source=path)
    detail.append(row)
detail = pd.DataFrame(detail)
for row in detail[detail.method.eq('TEEN')].itertuples():
    sota.loc[len(sota)] = [row.method, row.dataset, 'official-formula-transfer',
                           row.repeats, row.inc, row.all, row.auroc, row.f1]
for row in detail[detail.method.eq('FOWAC')].itertuples():
    provenance = 'proposed-formal-50'
    sota.loc[len(sota)] = [row.method, row.dataset, provenance,
                           row.repeats, row.inc, row.all, row.auroc, row.f1]

for col in ['inc_aa', 'all_aa', 'auroc_aa', 'osr_f1_aa']:
    sota[col] = (sota[col] * 100).map(lambda x: f'{x:.2f}')
for col in ['known', 'unknown', 'auroc', 'f1', 'aupr', 'fpr95', 'inc', 'all']:
    if col in detail:
        detail[col] = (detail[col] * 100).map(lambda x: f'{x:.2f}')

main_cols = ['method', 'dataset', 'provenance', 'repeats',
             'inc_aa', 'all_aa', 'auroc_aa', 'osr_f1_aa']
detail_cols = ['method', 'dataset', 'repeats', 'known', 'unknown', 'auroc',
               'aupr', 'fpr95', 'f1', 'inc', 'all', 'source']
protocol_only = pd.DataFrame([
    ['MetaGCD', 'original visual C-GCD protocol', 'not numerically mixed'],
    ['OCGCD/DEAN', 'original online visual C-GCD protocol', 'not numerically mixed'],
    ['OpenIncrement', 'original visual OSR+CIL protocol', 'not numerically mixed'],
    ['CAMP', 'ECCV 2024 visual GCCD protocol', 'not numerically mixed'],
    ['VB-CGCD', 'audio transfer attempted', 'DNF: singular 5-shot covariance'],
    ['FaE', 'AAAI 2026 visual C-GCD protocol', 'not numerically mixed'],
    ['PRISM', 'ICLR 2026 visual OW-CCD protocol', 'not numerically mixed'],
    ['OCCD', 'CVPR 2026 visual-drift protocol', 'not numerically mixed'],
    ['VC-CGCD', '2026 visual C-GCD preprint protocol', 'not numerically mixed'],
], columns=['method', 'provenance', 'status'])

def markdown(frame):
    values = [[str(x) for x in frame.columns]] + [
        [str(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(row[i]) for row in values) for i in range(len(values[0]))]
    def render(row):
        return '| ' + ' | '.join(cell.ljust(width) for cell, width in zip(row, widths)) + ' |'
    return '\n'.join([render(values[0]),
                      '| ' + ' | '.join('-' * width for width in widths) + ' |'] +
                     [render(row) for row in values[1:]])

content = [
    '# Current paper result tables', '',
    '> Generated from traceable local logs. FSC-89 FOWAC is withheld because the '
    'available audit has an invalid all-class evaluation; LS-100 is the completed '
    'frozen 50-repeat result.', '',
    '## Protocol-compatible main comparison (%)', '',
    markdown(sota[main_cols].sort_values(['dataset', 'method'])), '',
    '## Open-set recognition and continual metrics (%)', '',
    markdown(detail[detail_cols].sort_values(['dataset', 'method'])), '',
    '## Visual-domain methods retained as protocol-only references', '',
    markdown(protocol_only), '',
]
OUT.write_text('\n'.join(content))
print(f'wrote {OUT}')
