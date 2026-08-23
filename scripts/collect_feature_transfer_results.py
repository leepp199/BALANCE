#!/usr/bin/env python
"""Collect OFCL/OPCR and later feature-transfer baselines into one ledger."""
import csv
import json
import statistics
from pathlib import Path


roots = [Path('artifacts/ofcl_transfer'), Path('artifacts/opcr'), Path('artifacts/yloc'),
         Path('artifacts/happy')]
names = {'ls100': 'LS-100', 'ns100': 'NS-100', 'fsc89': 'FSC-89'}
rows = []
for root in roots:
    for path in sorted(root.glob('*/seed*/metrics.json')):
        data = json.loads(path.read_text())
        rec, inc = data['records'], data['records'][1:]
        cfg = data['config']
        rows.append({
            'method': data['method'], 'dataset': names[path.parts[-3]],
            'provenance': data['provenance'], 'seed': cfg['seed'],
            'base_acc': rec[0]['all_acc'],
            'inc_aa': sum(x['incremental_acc'] for x in inc) / len(inc),
            'final_inc': inc[-1]['incremental_acc'],
            'all_aa': sum(x['all_acc'] for x in inc) / len(inc),
            'auroc_aa': sum(x['auroc'] for x in inc) / len(inc),
            'osr_f1_aa': sum(x['osr_f1'] for x in inc) / len(inc),
            'source': str(path),
        })
out = Path('experiments/feature_transfer_three_datasets.csv')
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f'wrote {len(rows)} rows to {out}')

summary = []
for method in sorted({r['method'] for r in rows}):
    for dataset in names.values():
        group = [r for r in rows if r['method'] == method and r['dataset'] == dataset]
        if not group:
            continue
        record = {'method': method, 'dataset': dataset, 'n_seeds': len(group),
                  'repeat_unit': 'algorithm seed; frozen session features'}
        for metric in ('base_acc', 'inc_aa', 'final_inc', 'all_aa', 'auroc_aa', 'osr_f1_aa'):
            values = [float(r[metric]) for r in group]
            record[f'{metric}_mean'] = statistics.mean(values)
            record[f'{metric}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
summary_out = Path('experiments/feature_transfer_summary.csv')
with summary_out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader(); w.writerows(summary)
print(f'wrote {len(summary)} aggregate rows to {summary_out}')
