#!/usr/bin/env python
"""Build provenance-aware ICASSP result tables from completed experiment ledgers."""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
OUT = ROOT / 'docs' / 'generated_tables.md'


def pct(x):
    return '--' if pd.isna(x) else f'{100 * float(x):.2f}'


sections = ['# Generated experiment tables', '',
            '> Generated from CSV ledgers; audit and screening rows are excluded from main tables.', '']

sota_path = EXP / 'sota_three_datasets.csv'
feature_path = EXP / 'feature_transfer_summary.csv'
prop_path = EXP / 'proposed_three_datasets.csv'

main_rows = []
if sota_path.exists():
    s = pd.read_csv(sota_path)
    eligible = s[s.get('main_table_eligible', False).astype(str).str.lower().eq('true')]
    for _, r in eligible.iterrows():
        main_rows.append([r['method'], r['dataset'], pct(r.get('inc_aa')),
                          pct(r.get('final_inc')), pct(r.get('all_aa')), 'data-stream repeats'])
if feature_path.exists():
    f = pd.read_csv(feature_path)
    for _, r in f.iterrows():
        main_rows.append([r['method'], r['dataset'],
                          f"{pct(r['inc_aa_mean'])} ± {pct(r['inc_aa_std'])}",
                          f"{pct(r['final_inc_mean'])} ± {pct(r['final_inc_std'])}",
                          f"{pct(r['all_aa_mean'])} ± {pct(r['all_aa_std'])}",
                          f"algorithm seeds (n={int(r['n_seeds'])})"])
if prop_path.exists():
    p = pd.read_csv(prop_path)
    eligible = p[p['main_table_eligible'].astype(str).str.lower().eq('true')]
    for _, r in eligible.iterrows():
        main_rows.append([r['method'], r['dataset'], pct(r['inc_aa']),
                          f"{pct(r['final_inc'])} ± {pct(r['final_inc_std'])}",
                          pct(r['all_aa']), 'paired data-stream repeats'])

sections += ['## Main continual-learning comparison', '',
             '| Method | Dataset | AA-inc (%) | Final-inc (%) | AA-all (%) | Repeat unit |',
             '|---|---:|---:|---:|---:|---|']
sections += ['| ' + ' | '.join(map(str, row)) + ' |' for row in main_rows]

sig_path = EXP / 'paired_significance.csv'
if sig_path.exists():
    q = pd.read_csv(sig_path)
    q = q[(q.metric.isin(['inc', 'final_inc', 'all'])) & q.complete_10_pairs.astype(bool)]
    sections += ['', '## Paired significance versus TEEN', '',
                 '| Method | Dataset | Metric | Gain (pp) | 95% CI (pp) | paired-t p | Wilcoxon p |',
                 '|---|---|---|---:|---:|---:|---:|']
    for _, r in q.iterrows():
        sections.append('| {} | {} | {} | {:.2f} | [{:.2f}, {:.2f}] | {:.4g} | {:.4g} |'.format(
            r.method, r.dataset, r.metric, 100*r.mean_gain, 100*r.gain_ci95_low,
            100*r.gain_ci95_high, r.paired_t_p, r.wilcoxon_p))

extended_path = EXP / 'extended_metrics_summary.csv'
if extended_path.exists():
    e = pd.read_csv(extended_path)
    e = e[e.n_repeats.eq(10)]
    sections += ['', '## Metric-complete open-set and discovery evaluation', '',
                 '| Method | Dataset | AUPR (%) | FPR95 (%) ↓ | Clu-ACC (%) | NMI (%) | ARI (%) |',
                 '|---|---|---:|---:|---:|---:|---:|']
    for _, r in e.iterrows():
        sections.append('| {} | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} |'.format(
            r.method, r.dataset,
            100*r.aupr_mean, 100*r.aupr_std, 100*r.fpr95_mean, 100*r.fpr95_std,
            100*r.cluster_acc_mean, 100*r.cluster_acc_std, 100*r.nmi_mean,
            100*r.nmi_std, 100*r.ari_mean, 100*r.ari_std))

extended_sig_path = EXP / 'paired_extended_significance.csv'
if extended_sig_path.exists():
    x = pd.read_csv(extended_sig_path)
    x = x[x.complete_10_pairs.astype(bool)]
    if len(x):
        sections += ['', '## Paired extended-metric significance: FOWAC-BCD minus TEEN', '',
                     '| Metric | Gain (pp) | 95% CI (pp) | paired-t p | Wilcoxon p |',
                     '|---|---:|---:|---:|---:|']
        for _, r in x.iterrows():
            sections.append('| {} | {:.2f} | [{:.2f}, {:.2f}] | {:.4g} | {:.4g} |'.format(
                r.metric, 100*r.mean_gain, 100*r.gain_ci95_low,
                100*r.gain_ci95_high, r.paired_t_p, r.wilcoxon_p))

OUT.write_text('\n'.join(sections) + '\n')
print(f'wrote {OUT}')
