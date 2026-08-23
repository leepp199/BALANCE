#!/usr/bin/env python
"""Collect final summary blocks from three-dataset SOTA logs into paper-ready CSV."""
import argparse
import csv
import re
from pathlib import Path


FIELDS = {
    'known_aa': r'Average Acc Known:\s*([0-9.]+)',
    'novel_aa': r'Average Acc Unknown:\s*([0-9.]+)',
    'auroc_aa': r'Average AUROC:\s*([0-9.]+)',
    'f1_aa': r'Average F1 Score:\s*([0-9.]+)',
    'inc_aa': r'Average Incremental Acc:\s*([0-9.]+)',
    'all_aa': r'Average all Acc:\s*([0-9.]+)',
    'inc_pd': r'PD Incremental Acc:\s*([-0-9.]+)',
    'all_pd': r'PD all Acc:\s*([-0-9.]+)',
}


def parse(path):
    text = path.read_text(errors='replace')
    summaries = text.split('=== Sessions Average Accuracy (AA) ===')
    if len(summaries) < 2:
        return None
    tail = summaries[-1]
    row = {'source_log': str(path)}
    base = re.findall(r'=== Final Session 0 ===\s*Average Acc:\s*([0-9.]+)', text)
    row['base_acc'] = float(base[-1]) if base else ''
    for key, pattern in FIELDS.items():
        match = re.search(pattern, tail)
        row[key] = float(match.group(1)) if match else ''
    session4 = re.findall(r'total session4 incremental acc is\s*([0-9.]+)\s*±\s*([0-9.]+)', text)
    row['final_inc'] = float(session4[-1][0]) if session4 else ''
    row['final_inc_std'] = float(session4[-1][1]) if session4 else ''
    return row


def infer_identity(path):
    name = path.stem.lower()
    method = 'TEEN' if 'teen' in name else name.split('_')[0].upper()
    dataset = 'LS-100' if 'ls100' in name else 'NS-100' if 'ns100' in name else 'FSC-89' if 'fsc89' in name else 'unknown'
    return method, dataset


def infer_provenance(path):
    """Keep audit runs in the ledger without allowing them into the main table."""
    name = path.stem.lower()
    if name == 'teen_pure_ls100_10runs':
        return 'old-checkpoint-audit', False
    if 'canonical' in name:
        return 'canonical-checkpoint', True
    if name.startswith('teen_pure_'):
        return 'official-formula-transfer', True
    return 'adapted-or-screening', False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', default='logs/sota_three_datasets')
    p.add_argument('--output', default='experiments/sota_three_datasets.csv')
    args = p.parse_args()
    rows = []
    for path in sorted(Path(args.logs).glob('*.log')):
        row = parse(path)
        if row:
            row['method'], row['dataset'] = infer_identity(path)
            row['variant'], row['main_table_eligible'] = infer_provenance(path)
            rows.append(row)
    columns = ['method', 'dataset', 'variant', 'main_table_eligible', 'base_acc', *FIELDS,
               'final_inc', 'final_inc_std', 'source_log']
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} completed runs to {output}')


if __name__ == '__main__':
    main()
