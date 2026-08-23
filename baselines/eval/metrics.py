"""
Unified evaluation metrics for FOWAC baselines.
Computes: Inc_acc, All_acc, AA, PD, AUROC, FPR95, acc_known per session.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def cluster_acc(y_true, y_pred, return_ind=False):
    """Hungarian-algorithm based cluster accuracy."""
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    if return_ind:
        return acc, ind, w
    return acc


def compute_per_sample_metrics(all_gt, all_pred, all_is_unknown_gt, all_is_unknown_pred,
                                known_class_range, session_classes):
    """
    Compute per-session incremental and overall accuracy metrics.
    
    Args:
        all_gt: list of ground truth class labels per session
        all_pred: list of predicted class labels per session
        all_is_unknown_gt: list of bool arrays (True=unknown) per session
        all_is_unknown_pred: list of bool arrays (True=flagged as unknown) per session
        known_class_range: tuple (min, max) of known class indices
        session_classes: list of class sets per session [0:base, base:base+way, ...]
    
    Returns:
        dict with inc_acc, all_acc, acc_known, acc_unknown per session and averages.
    """
    num_sessions = len(all_gt)
    results = defaultdict(list)
    
    # All encountered classes by session end
    all_encountered = []
    cum_classes = set()
    
    for s in range(num_sessions):
        gt = np.array(all_gt[s])
        pred = np.array(all_pred[s])
        is_unknown_gt = np.array(all_is_unknown_gt[s])
        is_unknown_pred = np.array(all_is_unknown_pred[s])
        encountered = session_classes[s]
        cum_classes.update(encountered)
        all_encountered.append(set(encountered))
        
        n_total = len(gt)
        if n_total == 0:
            results['inc_acc'].append(0.0)
            results['all_acc'].append(0.0)
            results['acc_known'].append(0.0)
            results['acc_unknown'].append(0.0)
            results['n_known'].append(0)
            results['n_unknown'].append(0)
            continue
        
        n_known_gt = sum(~is_unknown_gt)
        n_unknown_gt = sum(is_unknown_gt)
        results['n_known'].append(n_known_gt)
        results['n_unknown'].append(n_unknown_gt)
        
        # Inc_acc: accuracy on NEW classes in this session
        if s == 0:
            new_classes = set(encountered)
        else:
            new_classes = set(encountered) - all_encountered[s-1]
        
        inc_gt_mask = np.isin(gt, list(new_classes)) & (~is_unknown_gt)
        if inc_gt_mask.sum() > 0:
            inc_correct = (gt[inc_gt_mask] == pred[inc_gt_mask]).sum()
            inc_acc = inc_correct / inc_gt_mask.sum()
        else:
            inc_acc = 0.0
        results['inc_acc'].append(inc_acc)
        
        # All_acc: accuracy on ALL encountered classes among known predictions
        known_mask = ~is_unknown_pred
        known_gt_actually_known = ~is_unknown_gt
        # Only evaluate known predictions against known ground truth
        eval_mask = known_mask & known_gt_actually_known
        if eval_mask.sum() > 0:
            all_correct = (gt[eval_mask] == pred[eval_mask]).sum()
            all_acc = all_correct / eval_mask.sum()
        else:
            all_acc = 0.0
        results['all_acc'].append(all_acc)
        
        # acc_known: known samples correctly classified as known
        if n_known_gt > 0:
            known_correct_mask = (~is_unknown_gt) & (~is_unknown_pred)
            acc_known = known_correct_mask.sum() / n_known_gt
        else:
            acc_known = 0.0
        results['acc_known'].append(acc_known)
        
        # acc_unknown: unknown samples correctly flagged as unknown
        if n_unknown_gt > 0:
            unknown_correct_mask = is_unknown_gt & is_unknown_pred
            acc_unknown = unknown_correct_mask.sum() / n_unknown_gt
        else:
            acc_unknown = 0.0
        results['acc_unknown'].append(acc_unknown)
    
    # Compute AA (average accuracy) and PD (prediction difficulty)
    inc_arr = np.array(results['inc_acc'])
    all_arr = np.array(results['all_acc'])
    known_arr = np.array(results['acc_known'])
    unknown_arr = np.array(results['acc_unknown'])
    
    results['AA_inc'] = inc_arr.mean()
    results['AA_all'] = all_arr.mean()
    results['AA_known'] = known_arr.mean()
    results['AA_unknown'] = unknown_arr.mean()
    results['PD_inc'] = 1.0 - results['AA_inc']  # higher PD = worse
    results['PD_all'] = 1.0 - results['AA_all']
    
    # Also weight by per-bucket if available
    # (0-79), (80-84), (85-89), (90-94), (95-99) difficulty bins
    difficulty_bins = [(0, 79), (80, 84), (85, 89), (90, 94), (95, 99)]
    bin_results = {}
    for lo, hi in difficulty_bins:
        pct_start = lo / 100.0
        pct_end = hi / 100.0
        bin_idx = int(lo / 5) if lo < 90 else int((lo - 90) / 5 + 3)
        if bin_idx < num_sessions:
            bin_results[f'{lo}-{hi}'] = {
                'inc_acc': float(inc_arr[bin_idx]),
                'all_acc': float(all_arr[bin_idx]),
                'acc_known': float(known_arr[bin_idx]),
                'acc_unknown': float(unknown_arr[bin_idx]),
            }
    
    return {
        'per_session': {k: [float(x) for x in v] for k, v in results.items() 
                       if k in ['inc_acc', 'all_acc', 'acc_known', 'acc_unknown']},
        'AA_inc': float(results['AA_inc']),
        'AA_all': float(results['AA_all']),
        'AA_known': float(results['AA_known']),
        'AA_unknown': float(results['AA_unknown']),
        'PD_inc': float(results['PD_inc']),
        'PD_all': float(results['PD_all']),
        'bins': bin_results,
    }


def compute_auroc_fpr95(known_scores, unknown_scores, known_labels, unknown_labels):
    """
    Compute AUROC and FPR95 for open-set detection.
    
    Args:
        known_scores: confidence scores for known samples (higher = more confident known)
        unknown_scores: confidence scores for unknown samples
        known_labels: ground truth (should be 0 for known)
        unknown_labels: ground truth (should be 1 for unknown)
    
    Returns:
        (auroc, fpr95)
    """
    all_scores = np.concatenate([np.array(known_scores), np.array(unknown_scores)])
    all_labels = np.concatenate([np.zeros_like(known_labels), np.ones_like(unknown_labels)])
    
    # AUROC (higher is better, inlier=1 means "known")
    # We use the score as confidence of being "known"
    # So known samples should have higher scores
    if len(np.unique(all_labels)) < 2:
        return 0.5, 1.0
    
    try:
        auroc = sk_metrics.roc_auc_score(all_labels, all_scores)
    except:
        auroc = 0.5
    
    # FPR95: false positive rate at 95% true positive rate
    # FPR when TPR = 0.95
    fpr, tpr, thresholds = sk_metrics.roc_curve(all_labels, all_scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]
    
    return float(auroc), float(fpr95)
