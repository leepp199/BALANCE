#!/usr/bin/env python
import csv
import json
from pathlib import Path


rows = []
for path in sorted(Path('artifacts/ofcl_transfer').glob('*/seed*/metrics.json')):
    data = json.loads(path.read_text())
    rec = data['records']
    inc = rec[1:]
    name = path.parts[-3]
    dataset = {'ls100': 'LS-100', 'ns100': 'NS-100', 'fsc89': 'FSC-89'}[name]
    rows.append({
        'method': 'OFCL-acoustic-transfer', 'dataset': dataset,
        'provenance': data['provenance'], 'seed': data['config']['seed'],
        'base_acc': rec[0]['all_acc'],
        'inc_aa': sum(x['incremental_acc'] for x in inc) / len(inc),
        'final_inc': inc[-1]['incremental_acc'],
        'all_aa': sum(x['all_acc'] for x in inc) / len(inc),
        'auroc_aa': sum(x['auroc'] for x in inc) / len(inc),
        'osr_f1_aa': sum(x['osr_f1'] for x in inc) / len(inc),
        'source': str(path),
    })

out = Path('experiments/ofcl_three_datasets.csv')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f'wrote {len(rows)} rows to {out}')
