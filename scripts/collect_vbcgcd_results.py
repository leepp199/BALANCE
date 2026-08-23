#!/usr/bin/env python
import csv
import json
from pathlib import Path


ROOT = Path('artifacts/vbcgcd_formal')
OUTPUT = Path('experiments/vbcgcd_three_datasets.csv')
DATASETS = {'ls100': 'LS-100', 'ns100': 'NS-100', 'fsc89': 'FSC-89'}


rows = []
for key, dataset in DATASETS.items():
    for path in sorted((ROOT / key).glob('*/metrics.json')):
        records = json.loads(path.read_text())
        if len(records) < 5:
            continue
        inc = records[1:]
        rows.append({
            'method': 'VB-CGCD official-transfer',
            'dataset': dataset,
            'seed': path.parent.name,
            'base_acc': records[0]['all_acc'],
            'old_aa': sum(r['old_acc'] for r in inc) / len(inc),
            'novel_aa': sum(r['novel_acc'] for r in inc) / len(inc),
            'inc_aa': sum(r['incremental_acc'] for r in inc) / len(inc),
            'all_aa': sum(r['all_acc'] for r in inc) / len(inc),
            'final_inc': records[-1]['incremental_acc'],
            'final_all': records[-1]['all_acc'],
            'status': 'numerically-collapsed' if records[-1]['all_acc'] < 0.05 else 'complete',
            'source': str(path),
        })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fields = ['method', 'dataset', 'seed', 'base_acc', 'old_aa', 'novel_aa', 'inc_aa',
          'all_aa', 'final_inc', 'final_all', 'status', 'source']
with OUTPUT.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f'wrote {len(rows)} rows to {OUTPUT}')
