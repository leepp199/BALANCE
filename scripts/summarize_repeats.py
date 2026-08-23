#!/usr/bin/env python3
"""Summarize [RAW] continual metrics with sample std and normal 95% CI."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

PATTERN = re.compile(
    r'\[RAW\] round=(\d+) session=(\d+).*?inc=([0-9.]+) all=([0-9.]+)')


def stats(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = 1.96 * std / np.sqrt(len(values)) if len(values) else float('nan')
    mean = float(values.mean())
    return {'n': int(len(values)), 'mean': mean, 'sample_std': std,
            'ci95_low': mean - half, 'ci95_high': mean + half}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [(int(r), int(s), float(i), float(a))
            for r, s, i, a in PATTERN.findall(Path(args.log).read_text(errors='replace'))]
    rounds = sorted(set(row[0] for row in rows))
    if any(sum(row[0] == repeat for row in rows) != 4 for repeat in rounds):
        raise RuntimeError('Every repeat must contain exactly four completed sessions')
    result = {'source': str(Path(args.log).resolve()), 'independent_repeats': len(rounds),
              'sessions': {}}
    for session in range(1, 5):
        selected = [row for row in rows if row[1] == session]
        result['sessions'][str(session)] = {
            'inc_acc': stats([row[2] for row in selected]),
            'all_acc': stats([row[3] for row in selected]),
        }
    repeat_inc = [np.mean([row[2] for row in rows if row[0] == repeat])
                  for repeat in rounds]
    repeat_all = [np.mean([row[3] for row in rows if row[0] == repeat])
                  for repeat in rounds]
    result['sessions_average'] = {'inc_acc': stats(repeat_inc),
                                  'all_acc': stats(repeat_all)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['sessions_average'], indent=2))


if __name__ == '__main__':
    main()
