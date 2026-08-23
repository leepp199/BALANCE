#!/usr/bin/env python
"""Collect publishable repeat/session statistics only from explicit RAW-v2 records."""
import csv
import re
from pathlib import Path

import numpy as np


LINE = re.compile(
    r"\[RAW\] round=(?P<round>\d+) session=(?P<session>[1-4]) "
    r"acc known=(?P<known>[0-9.]+) acc unknown=(?P<unknown>[0-9.]+) "
    r"auroc=(?P<auroc>[0-9.]+) f1=(?P<f1>[0-9.]+) "
    r"inc=(?P<inc>[0-9.]+) all=(?P<all>[0-9.]+)"
)

SPECS = [
    ("TEEN", "LS-100", "logs/raw_v2_teen_ls100_10runs.log"),
    ("TEEN", "NS-100", "logs/raw_v2_teen_ns100_10runs.log"),
    ("TEEN", "FSC-89", "logs/raw_v2_teen_fsc89_10runs.log"),
    ("FOWAC-UMR", "LS-100", "logs/raw_v2_umr_ls100_10runs.log"),
    ("FOWAC-DS", "LS-100", "logs/raw_v2_dfsb_ds_ls100_10runs.log"),
    ("FOWAC-BCD", "FSC-89", "logs/raw_v2_bcd_fsc89_10runs.log"),
]
METRICS = ("known", "unknown", "auroc", "f1", "inc", "all")


def parse_complete(path):
    text = Path(path).read_text(errors="replace") if Path(path).exists() else ""
    grouped = {}
    for match in LINE.finditer(text):
        record = match.groupdict()
        grouped.setdefault(int(record["round"]), []).append(record)
    result = []
    for round_id in sorted(grouped):
        block = grouped[round_id]
        if len(block) != 4 or [int(x["session"]) for x in block] != [1, 2, 3, 4]:
            continue
        result.append((round_id, block))
    return result


def stats(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else np.nan


def main():
    summary, sessions = [], []
    for method, dataset, path in SPECS:
        repeats = parse_complete(path)
        complete = len(repeats) == 10
        for metric in METRICS:
            aa = [np.mean([float(x[metric]) for x in block]) for _, block in repeats]
            final = [float(block[-1][metric]) for _, block in repeats]
            aa_mean, aa_std = stats(aa) if aa else (np.nan, np.nan)
            final_mean, final_std = stats(final) if final else (np.nan, np.nan)
            summary.append(dict(method=method, dataset=dataset, metric=metric,
                                n_complete_repeats=len(repeats), complete_10=complete,
                                aa_mean=aa_mean, aa_sample_std=aa_std,
                                final_mean=final_mean, final_sample_std=final_std,
                                source=path))
            for session in range(1, 5):
                values = [float(block[session - 1][metric]) for _, block in repeats]
                mean, std = stats(values) if values else (np.nan, np.nan)
                sessions.append(dict(method=method, dataset=dataset, metric=metric,
                                     session=session, n=len(values), mean=mean,
                                     sample_std=std, complete_10=complete, source=path))
    out = Path("experiments/raw_v2_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with Path("experiments/raw_v2_sessions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sessions[0]))
        writer.writeheader(); writer.writerows(sessions)
    for row in summary:
        if row["metric"] == "inc":
            print(f'{row["method"]:10s} {row["dataset"]:6s} n={row["n_complete_repeats"]:2d} '
                  f'AA={row["aa_mean"]:.6f}±{row["aa_sample_std"]:.6f} '
                  f'final={row["final_mean"]:.6f}±{row["final_sample_std"]:.6f}')


if __name__ == "__main__":
    main()
