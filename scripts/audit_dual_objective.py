#!/usr/bin/env python3
"""Audit every candidate on both incremental and all-class accuracy."""
from pathlib import Path
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
line=re.compile(r'\[RAW\].*?round=(?P<r>\d+).*?session=(?P<s>[1-4]).*?inc=(?P<inc>[0-9.]+).*?all=(?P<all>[0-9.]+)')
logs={
 'LS-100/FOWAC-formal50':'logs/ls100_rank25_late_frozen_50.log',
 'NS-100/FOWAC-formal50':'logs/ns_all_q50_cana_50final.log',
 'FSC-89/FOWAC-corrected':'logs/fsc89_frozen_corrected.log',
 'FSC-89/FOWAC-historical-audit':'logs/fsc89_5765_historical_audit.log',
 'FSC-89/FOWAC-mean-cos-screen':'logs/fsc_mean_cos_serial.log',
}
rows=[]
for name,path in logs.items():
 text=(ROOT/path).read_text(errors='replace') if (ROOT/path).exists() else ''
 d=pd.DataFrame([m.groupdict() for m in line.finditer(text)])
 if d.empty: continue
 for c in d: d[c]=pd.to_numeric(d[c])
 d=d.groupby('r').filter(lambda x: len(x)==4)
 q=d.groupby('r')[['inc','all']].mean()
 rows.append({'method':name,'repeats':len(q),'inc_mean':q.inc.mean(),'inc_std':q.inc.std(ddof=1),'all_mean':q['all'].mean(),'all_std':q['all'].std(ddof=1),'dual_ok':bool(q.inc.mean()>=.60 and q['all'].mean()>=.80)})
out=pd.DataFrame(rows)
out.to_csv(ROOT/'experiments/dual_objective_audit.csv',index=False)
print(out.to_string(index=False))
