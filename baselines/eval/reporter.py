"""
Table reporter for FOWAC baseline results.
Generates comparison tables in markdown/text format.
"""
import json
import os
import numpy as np
from collections import OrderedDict


def render_table(rows, headers, title="", fmt="{:.4f}"):
    """Render a simple markdown table.
    
    Args:
        rows: list of dicts with same keys as headers
        headers: list of column names
        title: table title
        fmt: float formatting
    """
    lines = []
    if title:
        lines.append(f"\n### {title}\n")
    
    # Header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
    
    for row in rows:
        cells = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                cells.append(fmt.format(v))
            elif v is None or v == "":
                cells.append("-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(lines)


def generate_main_table(results, output_path):
    """
    Generate the main comparison table: method vs dataset vs metrics.
    
    Args:
        results: dict of {method_name: {dataset_name: metrics_dict}}
        output_path: path to save the table
    """
    methods = list(results.keys())
    datasets = list(next(iter(results.values())).keys())
    
    lines = []
    lines.append("# FOWAC Baseline Comparison\n")
    
    # Table 1: Main results - AA_inc, AA_all, AA_known, AA_unknown, PD, AUROC
    headers = ["Method"] + [f"{ds} AA_inc↑" for ds in datasets] + \
              [f"{ds} AA_all↑" for ds in datasets] + \
              [f"{ds} AA_known↑" for ds in datasets] + \
              [f"{ds} AA_unknown↑" for ds in datasets] + \
              [f"{ds} PD↓" for ds in datasets]
    
    lines.append("## Table 1: Overall Metrics\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
    
    for method in methods:
        row = [method]
        for ds in datasets:
            m = results[method][ds]
            row.append(f"{m.get('AA_inc', 0):.4f}")
        for ds in datasets:
            m = results[method][ds]
            row.append(f"{m.get('AA_all', 0):.4f}")
        for ds in datasets:
            m = results[method][ds]
            row.append(f"{m.get('AA_known', 0):.4f}")
        for ds in datasets:
            m = results[method][ds]
            row.append(f"{m.get('AA_unknown', 0):.4f}")
        for ds in datasets:
            m = results[method][ds]
            row.append(f"{m.get('PD_inc', 0):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("\n")
    
    # Table 2: Per-session breakdown
    lines.append("## Table 2: Per-Session Incremental Accuracy\n")
    
    for ds in datasets:
        n_sessions = len(results[methods[0]][ds].get('per_session', {}).get('inc_acc', []))
        if n_sessions == 0:
            continue
        
        lines.append(f"\n### Dataset: {ds}\n")
        headers = ["Method"] + [f"S{s}" for s in range(n_sessions)] + ["AA"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
        
        for method in methods:
            ps = results[method][ds].get('per_session', {})
            inc_accs = ps.get('inc_acc', [])
            row = [method] + [f"{a:.4f}" for a in inc_accs] + [f"{np.mean(inc_accs):.4f}" if inc_accs else "0.0000"]
            lines.append("| " + " | ".join(row) + " |")
    
    lines.append("\n")
    
    # Table 3: Per-Session Known Accuracy
    lines.append("## Table 3: Per-Session Known vs Unknown Detection\n")
    
    for ds in datasets:
        n_sessions = len(results[methods[0]][ds].get('per_session', {}).get('inc_acc', []))
        if n_sessions == 0:
            continue
        
        lines.append(f"\n### Dataset: {ds}\n")
        headers = ["Method"] + [f"S{s} Known" for s in range(n_sessions)] + ["AA_known"] + \
                  [f"S{s} Unk" for s in range(n_sessions)] + ["AA_unknown"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
        
        for method in methods:
            ps = results[method][ds].get('per_session', {})
            known_accs = ps.get('acc_known', [])
            unk_accs = ps.get('acc_unknown', [])
            row = [method] + [f"{a:.4f}" for a in known_accs] + [f"{np.mean(known_accs):.4f}" if known_accs else "0.0000"] + \
                  [f"{a:.4f}" for a in unk_accs] + [f"{np.mean(unk_accs):.4f}" if unk_accs else "0.0000"]
            lines.append("| " + " | ".join(row) + " |")
    
    lines.append("\n")
    
    # Table 4: Difficulty-binned results
    lines.append("## Table 4: Difficulty-Binned Incremental Accuracy\n")
    
    for method in methods:
        for ds in datasets:
            bins = results[method][ds].get('bins', {})
            if not bins:
                continue
            lines.append(f"\n### {method} on {ds}\n")
            headers = ["Bin"] + ["Inc_acc", "All_acc", "Acc_known", "Acc_unknown"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
            for bin_name, bm in bins.items():
                row = [bin_name, f"{bm['inc_acc']:.4f}", f"{bm['all_acc']:.4f}", 
                       f"{bm['acc_known']:.4f}", f"{bm['acc_unknown']:.4f}"]
                lines.append("| " + " | ".join(row) + " |")
    
    content = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Tables saved to {output_path}")
    return content


def generate_summary_json(results, output_path):
    """Save all results as JSON for easy post-processing."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Summary JSON saved to {output_path}")
