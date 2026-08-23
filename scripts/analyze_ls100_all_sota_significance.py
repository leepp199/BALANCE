#!/usr/bin/env python3
"""Significance ledger for all requested LS-100 SOTA names."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = re.compile(r'\[RAW\].*?round=(?P<round>\d+).*?session=(?P<s>[1-4]).*?inc=(?P<inc>[0-9.]+).*?all=(?P<all>[0-9.]+)')
rows = [m.groupdict() for m in RAW.finditer((ROOT/'logs/ls100_fowac_direct_frozen_50.log').read_text(errors='replace'))]
d = pd.DataFrame(rows)
for c in d: d[c] = pd.to_numeric(d[c])
d = d.groupby('round').filter(lambda x: len(x) == 4)
f = d.groupby('round')[['inc','all']].mean()
transfer = pd.read_csv(ROOT/'experiments/feature_transfer_three_datasets.csv')
transfer = transfer[transfer.dataset.eq('LS-100')]
requested = ['TEEN','OFCL','OPCR','YLOC','Happy','CAMP','MetaGCD','OCGCD/DEAN','OpenIncrement','VB-CGCD','FaE','PRISM','OCCD','VC-CGCD']
name_map = {'OFCL':'OFCL-acoustic-transfer','OPCR':'OPCR-acoustic','YLOC':'YLOC-PAL-acoustic','Happy':'Happy-HAProto-acoustic'}
out = []
rng = np.random.default_rng(20260823)
for method in requested:
    if method in name_map:
        z = transfer[transfer.method.eq(name_map[method])]
        base = z.inc_aa.to_numpy(float); prop = f.inc.to_numpy(float)
        boot = np.array([rng.choice(prop,len(prop),True).mean()-rng.choice(base,len(base),True).mean() for _ in range(20000)])
        out.append({'method':method,'metric':'inc','status':'estimable-independent-seeds','n_method':len(base),'n_fowac':len(prop),'fowac_minus_method':prop.mean()-base.mean(),'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),'welch_p':stats.ttest_ind(prop,base,equal_var=False).pvalue,'mannwhitney_p':stats.mannwhitneyu(prop,base,alternative='two-sided').pvalue})
    elif method == 'TEEN':
        q=pd.read_csv(ROOT/'experiments/fowac_paired_significance.csv'); q=q[(q.dataset=='LS-100')&(q.metric=='inc')].iloc[0]
        out.append({'method':method,'metric':'inc','status':'estimable-paired','n_method':10,'n_fowac':10,'fowac_minus_method':q.mean_gain,'ci_low':q.gain_ci95_low,'ci_high':q.gain_ci95_high,'welch_p':q.paired_t_p,'mannwhitney_p':q.wilcoxon_p})
    else:
        out.append({'method':method,'metric':'inc','status':'not-estimable: no LS-100 acoustic result','n_method':0,'n_fowac':len(f),'fowac_minus_method':np.nan,'ci_low':np.nan,'ci_high':np.nan,'welch_p':np.nan,'mannwhitney_p':np.nan})
out=pd.DataFrame(out)
out.to_csv(ROOT/'experiments/ls100_all_sota_significance.csv',index=False)
(ROOT/'docs/LS100_ALL_SOTA_SIGNIFICANCE.md').write_text(
    '# LS-100 all-requested-SOTA significance\n\n'
    'A method is estimable only when an LS-100 acoustic result exists. Visual-only methods are explicitly marked, not assigned fabricated values.\n\n'
    '```text\n' + out.to_string(index=False) + '\n```\n')
print(out.to_string(index=False))

# One compact paper-facing panel: estimable effects with CIs, plus explicit
# crossed markers for requested methods lacking LS-100 acoustic observations.
order = requested[::-1]
fig, ax = plt.subplots(figsize=(7.0, 4.2))
for y, method in enumerate(order):
    q = out[out.method.eq(method)].iloc[0]
    if q.status.startswith('not-estimable'):
        ax.plot(0, y, marker='x', ms=7, mew=1.5, color='#9CA3AF')
        ax.text(0.8, y, 'not estimable (no LS-100 acoustic result)', va='center', fontsize=7, color='#6B7280')
    else:
        gain, lo, hi = q.fowac_minus_method*100, q.ci_low*100, q.ci_high*100
        ax.errorbar(gain, y, xerr=[[gain-lo],[hi-gain]], fmt='o', ms=4,
                    color='#0072B2', ecolor='#6B7280', capsize=2, lw=.9)
        ax.text(hi+1.0, y, f'p={q.welch_p:.2g}', va='center', fontsize=7)
ax.axvline(0, color='#B91C1C', ls='--', lw=.8)
ax.set_yticks(range(len(order)), order)
ax.set_xlabel('FOWAC − method incremental accuracy (percentage points; 95% bootstrap CI)')
ax.set_title('LS-100 significance coverage for all requested SOTA')
ax.grid(axis='x', color='#D1D5DB', lw=.5, alpha=.75)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(ROOT/'figures'/f'ls100_all_sota_significance.{ext}', dpi=400, bbox_inches='tight')
plt.close(fig)
