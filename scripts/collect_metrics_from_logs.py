#!/usr/bin/env python3
import argparse
import csv
import os
import re

PATTERNS = {
    "s0_acc": re.compile(r"Average Acc:\s*([0-9.\-]+)"),
    "aa_known": re.compile(r"Average Acc Known:\s*([0-9.\-]+)"),
    "aa_unknown": re.compile(r"Average Acc Unknown:\s*([0-9.\-]+)"),
    "aa_f1": re.compile(r"Average F1 Score:\s*([0-9.\-]+)"),
    "aa_inc": re.compile(r"Average Incremental Acc:\s*([0-9.\-]+)"),
    "aa_all": re.compile(r"Average all Acc:\s*([0-9.\-]+)"),
    "pd_known": re.compile(r"PD Acc Known:\s*([0-9.\-]+)"),
    "pd_unknown": re.compile(r"PD Acc Unknown:\s*([0-9.\-]+)"),
    "pd_f1": re.compile(r"PD F1 Score:\s*([0-9.\-]+)"),
    "pd_inc": re.compile(r"PD Incremental Acc:\s*([0-9.\-]+)"),
    "pd_all": re.compile(r"PD all Acc:\s*([0-9.\-]+)"),
}


def extract(pattern, text):
    m = pattern.search(text)
    return m.group(1) if m else ""


def parse_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    row = {"file": os.path.basename(path)}
    for k, p in PATTERNS.items():
        row[k] = extract(p, txt)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="/data/lqq/baseline/save_result")
    ap.add_argument("--prefix", default="test_result")
    ap.add_argument("--output_csv", default="/data/lqq/baseline/save_result/metrics_summary.csv")
    args = ap.parse_args()

    files = [
        os.path.join(args.input_dir, n)
        for n in sorted(os.listdir(args.input_dir))
        if n.startswith(args.prefix) and n.endswith(".txt")
    ]

    rows = [parse_file(p) for p in files]

    cols = [
        "file", "s0_acc", "aa_known", "aa_unknown", "aa_f1", "aa_inc", "aa_all",
        "pd_known", "pd_unknown", "pd_f1", "pd_inc", "pd_all"
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"parsed={len(rows)} output={args.output_csv}")


if __name__ == "__main__":
    main()
