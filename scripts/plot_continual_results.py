#!/usr/bin/env python
"""Publication-ready per-session curves from authoritative repeated-run logs."""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
LINE = re.compile(r'\[RAW\] round=(?P<round>\d+) session=(?P<s>[1-4]).*?inc=(?P<inc>[0-9.]+) all=(?P<all>[0-9.]+)')


def blocks(path):
    paths = path if isinstance(path, (tuple, list)) else (path,)
    texts = []
    for index, item in enumerate(paths):
        found_text = Path(item).read_text(errors='replace')
        # The main 50-repeat process was stopped after its first 30 complete
        # repeats; exclude its incomplete next repeat. Parallel parts are
        # complete 10-repeat logs.
        if index == 0 and len(paths) > 1:
            raw = list(LINE.finditer(found_text))
            found_text = ''.join(m.group(0) + '\n' for m in raw[:120])
        texts.append(found_text)
    found = [(int(m['s']), float(m['inc']), float(m['all']))
             for text in texts for m in LINE.finditer(text)]
    result = []
    for i in range(0, len(found), 4):
        block = found[i:i + 4]
        if len(block) == 4 and [x[0] for x in block] == [1, 2, 3, 4]:
            result.append(np.asarray([[x[1], x[2]] for x in block]))
    return np.asarray(result)


datasets = {
    # Paper-facing names denote complete methods.  Internal modules (LSRB,
    # CANA, PAN and UMR) belong in the ablation figure, not in this legend.
    'LS-100': (('TEEN (baseline)', 'logs/raw_v2_teen_ls100_10runs.log'),
               ('FOWAC (ours)', 'logs/ls100_rank25_late_frozen_50.log')),
    'NS-100': (('TEEN (baseline)', 'logs/raw_v2_teen_ns100_10runs.log'),
               ('FOWAC (ours)', 'logs/ns_all_q50_cana_50final.log')),
    'FSC-89': (('TEEN (baseline)', 'logs/raw_v2_teen_fsc89_10runs.log'),
               ('FOWAC (ours)', (
                   'logs/fsc89_cosine_bias_-0p04_bankTrue_top3_clusterallFalse_ogateFalse_qgateFalse_qtop1_support_margin_sinkFalse_class_t0.05_reflow0.2.log',
                   'logs/fsc89_cosine_bias_-0p04_bankTrue_top3_clusterallFalse_ogateFalse_qgateFalse_qtop1_support_margin_sinkFalse_class_t0.05_reflow0.2_p1.log',
                   'logs/fsc89_cosine_bias_-0p04_bankTrue_top3_clusterallFalse_ogateFalse_qgateFalse_qtop1_support_margin_sinkFalse_class_t0.05_reflow0.2_p2.log'))),
}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.15), sharex=True)
colors = {'TEEN (baseline)': '#6B7280', 'FOWAC (ours)': '#0072B2'}
x = np.arange(1, 5)
for col, (dataset, methods) in enumerate(datasets.items()):
    for label, path in methods:
        resolved = tuple(ROOT / item for item in path) if isinstance(path, (tuple, list)) else ROOT / path
        values = blocks(resolved)
        if not len(values):
            continue
        for row, metric in enumerate((0, 1)):
            mean = values[:, :, metric].mean(0) * 100
            sem = values[:, :, metric].std(0, ddof=1) / np.sqrt(len(values)) * 100 if len(values) > 1 else np.zeros(4)
            axes[row, col].plot(x, mean, marker='o', ms=3, lw=1.4,
                                color=colors[label], label=label)
            axes[row, col].fill_between(x, mean - 1.96*sem, mean + 1.96*sem,
                                        color=colors[label], alpha=0.14, linewidth=0)
    axes[0, col].set_title(dataset, pad=4, weight='semibold')
    axes[1, col].set_xlabel('Incremental session')
    for row in range(2):
        axes[row, col].set_xticks(x)
        axes[row, col].grid(axis='y', color='#D1D5DB', lw=0.5, alpha=0.7)
        axes[row, col].spines[['top', 'right']].set_visible(False)
axes[0, 0].set_ylabel('Incremental accuracy (%)')
axes[1, 0].set_ylabel('All-class accuracy (%)')
legend = {}
for axis in axes.flat:
    for handle, label in zip(*axis.get_legend_handles_labels()):
        legend[label] = handle
labels = list(legend)
handles = [legend[label] for label in labels]
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.985),
           ncol=3, frameon=False)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.84,
                    wspace=0.26, hspace=0.13)
for suffix in ('pdf', 'png'):
    fig.savefig(OUT / f'continual_session_curves.{suffix}', dpi=400, bbox_inches='tight')
print(f'wrote {OUT / "continual_session_curves.pdf"}')
