#!/usr/bin/env python3
"""Render FOWAC tables in the structure of the supplied manuscript."""
from pathlib import Path
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
rx=re.compile(r'\[RAW\].*?round=(?P<r>\d+).*?session=(?P<s>[1-4]).*?acc known=(?P<known>[0-9.]+).*?acc unknown=(?P<unknown>[0-9.]+).*?auroc=(?P<auroc>[0-9.]+).*?f1=(?P<f1>[0-9.]+).*?inc=(?P<inc>[0-9.]+).*?all=(?P<all>[0-9.]+)')
logs={'LS-100':'logs/ls100_rank25_late_frozen_50.log','NS-100':'logs/ns_all_q50_cana_50final.log'}
frames={}
base_acc={'LS-100':0.9367,'NS-100':0.9985}
for ds,path in logs.items():
 d=pd.DataFrame([m.groupdict() for m in rx.finditer((ROOT/path).read_text(errors='replace'))])
 for c in d: d[c]=pd.to_numeric(d[c])
 d=d.groupby('r').filter(lambda x: len(x)==4)
 frames[ds]=d

def row(ds, metric):
 d=frames[ds].groupby('s')[metric].mean()
 return [f'{d.get(i,float("nan"))*100:.2f}' for i in range(1,5)] + [f'{d.mean()*100:.2f}']

lines=['# FOWAC results in the manuscript table format','',
       '> The session/AA tables follow the supplied document. FSC-89 is intentionally omitted from the main claim until its all-class evaluator is repaired; its current audit has all_acc below 10%.','']
for ds in logs:
 lines += [f'## Table: FOWAC on {ds} (%)','',
       '| Method | Metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA* |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
 for method in ['FOWAC (ours)']:
  for metric,name in [('inc','Inc_acc'),('all','All_acc')]:
   vals=row(ds,metric)
   if metric == 'all': vals=[f'{base_acc[ds]*100:.2f}',*vals]
   else: vals=['N/A',*vals]
   lines.append('| '+ ' | '.join([method,name,*vals])+' |')
 lines += ['']
lines += ['## Additional open-set table (%)','',
       '| Dataset | Metric | Session 0 | Session 1 | Session 2 | Session 3 | Session 4 | AA* |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for ds in logs:
 for metric,name in [('auroc','AUROC'),('f1','F1-score'),('known','Known accuracy'),('unknown','Unknown accuracy')]:
  vals=row(ds,metric)
  vals=['N/A',*vals]
  lines.append('| '+ ' | '.join([ds,name,*vals])+' |')
lines += ['', '*AA is the mean of incremental Sessions 1–4, matching the supplied manuscript. Session 0 is the base-session all-class accuracy; N/A metrics are undefined before open-world increments.']
(ROOT/'docs/FOWAC_RESULTS_DOC_STYLE.md').write_text('\n'.join(lines)+'\n')
print('wrote docs/FOWAC_RESULTS_DOC_STYLE.md')
