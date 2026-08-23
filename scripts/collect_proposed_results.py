#!/usr/bin/env python
"""Collect completed repeated proposed-method runs and retain screening ablations."""
import csv
from pathlib import Path

from collect_sota_results import FIELDS, parse


logs = Path('logs')
paths = sorted(logs.glob('proposed_statmem*.log'))
rows = []
for path in paths:
    row = parse(path)
    if row is None:
        continue
    name = path.stem.lower()
    row['method'] = 'FOWAC-UMR'
    # Do not use a raw ``'ns' in name`` test: the suffix ``10runs`` contains
    # ``ns`` and previously mislabeled FSC-89 rows as NS-100.
    row['dataset'] = ('FSC-89' if 'fsc' in name
                      else 'NS-100' if 'ns100' in name or '_ns_' in name
                      else 'LS-100')
    row['variant'] = ('full-writeback-negative-ablation' if name == 'proposed_statmem_ls_screen'
                      else 'checkpoint-or-state-drift-audit' if 'audit' in name
                      else 'auto-variance-rng-shift-audit' if name == 'proposed_statmem_auto_ls_screen'
                      else 'cached-variance-screen' if 'cachedvar' in name and 'screen' in name
                      else 'cached-variance-ungated-ablation' if name == 'proposed_statmem_cachedvar_ns100_10runs'
                      else 'cached-variance-main' if 'cachedvar' in name and '10runs' in name
                      else 'residual-screen' if 'screen' in name
                      else 'fixed-variance-ablation')
    row['main_table_eligible'] = row['variant'] == 'cached-variance-main'
    rows.append(row)

# Structure-referenced mixed-stream runs predate the ``proposed_`` filename
# convention. Include only the leakage-safe ten-repeat logs here; novel-only
# structure screens remain audit evidence and are deliberately excluded.
for path, variant, eligible in [
    (logs / 'mixed_openworld_dual05_10runs.log', 'dfsb-dual-space-weight-0.5', True),
    (logs / 'mixed_openworld_dual15_10runs.log', 'dfsb-dual-space-weight-1.5', False),
]:
    row = parse(path)
    if row is None:
        continue
    row['method'] = 'FOWAC-DS'
    row['dataset'] = 'LS-100'
    row['variant'] = variant
    row['main_table_eligible'] = eligible
    rows.append(row)

balanced_path = logs / 'proposed_balanced_kmeans_fsc89_10runs.log'
balanced_row = parse(balanced_path)
if balanced_row is not None:
    balanced_row['method'] = 'FOWAC-BCD'
    balanced_row['dataset'] = 'FSC-89'
    balanced_row['variant'] = 'balanced-capacity-discovery'
    balanced_row['main_table_eligible'] = True
    rows.append(balanced_row)

columns = ['method', 'dataset', 'variant', 'main_table_eligible', 'base_acc', *FIELDS,
           'final_inc', 'final_inc_std', 'source_log']
out = Path('experiments/proposed_three_datasets.csv')
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader(); writer.writerows(rows)
print(f'wrote {len(rows)} completed runs to {out}')
