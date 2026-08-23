import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np


def parse_result_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    session_pattern = re.compile(
        r"session:\s*(\d+),\s*total aac known:\s*([0-9.]+)\s*±\s*([0-9.]+),\s*"
        r"total acc unknown:\s*([0-9.]+)\s*±\s*([0-9.]+),\s*"
        r"total f1 score:\s*([0-9.]+)\s*±\s*([0-9.]+),\s*"
        r"total incremental acc:\s*([0-9.]+)\s*±\s*([0-9.]+),\s*"
        r"total all acc:\s*([0-9.]+)\s*±\s*([0-9.]+)"
    )

    sessions = []
    known = []
    unknown = []
    f1 = []
    inc = []
    all_acc = []

    for m in session_pattern.finditer(text):
        sessions.append(int(m.group(1)))
        known.append(float(m.group(2)))
        unknown.append(float(m.group(4)))
        f1.append(float(m.group(6)))
        inc.append(float(m.group(8)))
        all_acc.append(float(m.group(10)))

    aa = {}
    for k, pat in {
        'AA_Known': r"Average Acc Known:\s*([0-9.\-]+)",
        'AA_Unknown': r"Average Acc Unknown:\s*([0-9.\-]+)",
        'AA_F1': r"Average F1 Score:\s*([0-9.\-]+)",
        'AA_Incremental': r"Average Incremental Acc:\s*([0-9.\-]+)",
        'AA_All': r"Average all Acc:\s*([0-9.\-]+)",
        'PD_Known': r"PD Acc Known:\s*([0-9.\-]+)",
        'PD_Unknown': r"PD Acc Unknown:\s*([0-9.\-]+)",
        'PD_F1': r"PD F1 Score:\s*([0-9.\-]+)",
        'PD_Incremental': r"PD Incremental Acc:\s*([0-9.\-]+)",
        'PD_All': r"PD all Acc:\s*([0-9.\-]+)",
    }.items():
        mm = re.search(pat, text)
        aa[k] = float(mm.group(1)) if mm else np.nan

    return {
        'sessions': sessions,
        'known': known,
        'unknown': unknown,
        'f1': f1,
        'inc': inc,
        'all': all_acc,
        'summary': aa,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='/data/lqq/baseline/save_result/plots')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = parse_result_file(args.result)

    x = data['sessions']
    if len(x) == 0:
        raise RuntimeError('No session summary rows found in result file.')

    plt.figure(figsize=(8, 5))
    plt.plot(x, data['known'], marker='o', label='Known Acc')
    plt.plot(x, data['unknown'], marker='s', label='Unknown Acc')
    plt.plot(x, data['f1'], marker='^', label='F1')
    plt.plot(x, data['inc'], marker='d', label='Incremental Acc')
    plt.plot(x, data['all'], marker='*', label='All Acc')
    plt.xlabel('Session')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.title('Open-World FSCIL Session Curves')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(args.outdir, 'session_curves.png')
    plt.savefig(curve_path, dpi=220)
    plt.close()

    keys_aa = ['AA_Known', 'AA_Unknown', 'AA_F1', 'AA_Incremental', 'AA_All']
    vals_aa = [data['summary'][k] for k in keys_aa]
    plt.figure(figsize=(8, 4.8))
    plt.bar(keys_aa, vals_aa)
    plt.ylim(0, 1.05)
    plt.title('AA Metrics')
    plt.xticks(rotation=20)
    plt.tight_layout()
    aa_path = os.path.join(args.outdir, 'aa_bar.png')
    plt.savefig(aa_path, dpi=220)
    plt.close()

    keys_pd = ['PD_Known', 'PD_Unknown', 'PD_F1', 'PD_Incremental', 'PD_All']
    vals_pd = [data['summary'][k] for k in keys_pd]
    plt.figure(figsize=(8, 4.8))
    plt.bar(keys_pd, vals_pd)
    plt.title('PD Metrics (Session1 - SessionLast)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    pd_path = os.path.join(args.outdir, 'pd_bar.png')
    plt.savefig(pd_path, dpi=220)
    plt.close()

    summary_txt = os.path.join(args.outdir, 'summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        for k, v in data['summary'].items():
            f.write(f'{k}: {v:.4f}\n')

    print('Saved:', curve_path)
    print('Saved:', aa_path)
    print('Saved:', pd_path)
    print('Saved:', summary_txt)


if __name__ == '__main__':
    main()
