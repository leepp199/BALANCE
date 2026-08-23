import os

import torch.nn.functional as F
import torch
from utils.utils import count_acc
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def _adaptive_margin_threshold(margins: torch.Tensor) -> float:
    """
    使用批内双簇聚类自适应估计阈值，避免手工调参。
    已知样本通常 margin 更大，未知样本 margin 更小。
    """
    m = margins.detach().view(-1).cpu().numpy()
    if m.size == 0:
        return 0.0
    if m.size < 4 or float(np.std(m)) < 1e-8:
        return float(np.median(m))

    km = KMeans(n_clusters=2, n_init=20, random_state=42)
    km.fit(m.reshape(-1, 1))
    c0, c1 = float(km.cluster_centers_[0, 0]), float(km.cluster_centers_[1, 0])
    return (c0 + c1) / 2.0


def _support_calibrated_thresholds(
    margins: torch.Tensor,
    pos_scores: torch.Tensor,
    support_count: int,
) -> tuple:
    margin_thr = _adaptive_margin_threshold(margins)
    pos_thr = float(torch.median(pos_scores).item())

    if support_count <= 0:
        return margin_thr, pos_thr

    support_count = min(support_count, margins.numel())
    support_margin = margins[:support_count].detach()
    support_pos = pos_scores[:support_count].detach()

    if support_margin.numel() > 0:
        q_margin = torch.quantile(support_margin, 0.10).item()
        margin_thr = max(margin_thr, float(q_margin))

    if support_pos.numel() > 0:
        q_pos = torch.quantile(support_pos, 0.10).item()
        pos_thr = max(pos_thr, float(q_pos))

    return margin_thr, pos_thr

#baseline1 
# Task 1.3: 加入 session-aware 平滑，跨 session 保留 margin/cls_margin 分位点
_SESSION_STATS = {
    'margins': [],        # running quantile list per session
    'cls_margins': [],
}


def _session_aware_threshold(margins_tensor, cls_margins_tensor, session,
                              blend: float = 0.0):
    """
    Session-aware 阈值混合 (Task 1.3 修订)：
    默认 blend=0 表示完全使用每批自适应阈值（与 opt_v3 基线一致）。
    当 blend>0 时，对 session>=3 施加历史 quantile 的轻度 shrinkage：
        thr = (1-blend)*cur + blend*hist
    实测 blend=0.5 会把已知样本误判为未知，导致 known_acc 崩盘，因此
    默认关闭，只保留统计累计以便后续 ablation。
    """
    thr_open = _adaptive_margin_threshold(margins_tensor)
    thr_cls = float(torch.quantile(cls_margins_tensor, 0.35).item())

    if session is not None and session <= 2:
        _SESSION_STATS['margins'].append(float(torch.quantile(margins_tensor, 0.5).item()))
        _SESSION_STATS['cls_margins'].append(float(torch.quantile(cls_margins_tensor, 0.35).item()))
    elif blend > 0 and session is not None and session >= 3 and _SESSION_STATS['margins']:
        hist_open = float(np.mean(_SESSION_STATS['margins']))
        hist_cls = float(np.mean(_SESSION_STATS['cls_margins']))
        thr_open = (1.0 - blend) * thr_open + blend * hist_open
        thr_cls = (1.0 - blend) * thr_cls + blend * hist_cls
    return thr_open, thr_cls


def reset_session_stats():
    """每次评测开始时调用，避免跨 test_time / run 的历史污染。"""
    _SESSION_STATS['margins'].clear()
    _SESSION_STATS['cls_margins'].clear()


def _unknown_auroc(labels, known_margin, num_known_classes):
    """AUROC with unknown samples as the positive class.

    ``known_margin`` is larger for known samples (positive-prototype score minus
    negative/open-prototype score), so its sign must be flipped before passing it
    to sklearn where a larger score is expected for the positive class.
    """
    labels = np.asarray(labels)
    known_margin = np.asarray(known_margin, dtype=np.float64)
    unknown_target = (labels >= int(num_known_classes)).astype(np.int64)
    if np.unique(unknown_target).size < 2:
        return float('nan')
    return float(roc_auc_score(unknown_target, -known_margin))


def _unknown_metrics(labels, known_margin, num_known_classes):
    labels = np.asarray(labels)
    score = -np.asarray(known_margin, dtype=np.float64)
    target = (labels >= int(num_known_classes)).astype(np.int64)
    if np.unique(target).size < 2:
        return {'auroc': float('nan'), 'aupr': float('nan'), 'fpr95': float('nan')}
    fpr, tpr, _ = roc_curve(target, score)
    valid = fpr[tpr >= 0.95]
    return {'auroc': float(roc_auc_score(target, score)),
            'aupr': float(average_precision_score(target, score)),
            'fpr95': float(valid.min()) if len(valid) else 1.0}


def run_test_fsl(model, args, test_loader, session=None):
    unknowns, unlabels, knowns, klabels = [], [], [], []
    accepted_boundary_distance = []
    session_labels, session_known_margins = [], []
    ranked_novel_candidates = []
    # 获取 N 个正原型
    proto = model.fc.weight[:args.num_labeled_classes, :].detach()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with tqdm(test_loader, total=len(test_loader), leave=False) as pbar:  
        for idx, batch in enumerate(pbar):
            data, label = [_.cuda() for _ in batch]
            data, label = data.squeeze(), label.squeeze()
            
            # Use the same current encoder for rejection and support prototypes.
            feature = model.encode(data)
            feature = feature.to(device)
            
            # 2. 计算分数  (UOP: 单一全局开集原型 → scores [M, N+1])
            use_uop = bool(getattr(args, 'use_uop', False))
            uop_chunk = int(getattr(args, 'uop_chunk_size', 5))
            uop_mode = str(getattr(args, 'uop_mode', 'antimean'))
            osr_noise = float(getattr(args, 'osr_noise_std', 0.1))
            scores = compute_feats(
                model, label[:args.n_ways*5], feature, proto,
                use_uop=use_uop, uop_chunk_size=uop_chunk, uop_mode=uop_mode,
                osr_noise_std=osr_noise,
            )

            # 3. 拆分正负分数
            num_cls = args.num_labeled_classes
            positive_scores = scores[:, :num_cls]
            if use_uop:
                # UOP 路径: 负分数是单一全局开集原型分数, 所有 query 共享
                uop_score = scores[:, num_cls]                              # [M]
                pred_cls_idx = torch.argmax(positive_scores, dim=1)
                pos_score = positive_scores.gather(1, pred_cls_idx.unsqueeze(1)).squeeze(1)
                neg_score = uop_score
            else:
                negative_scores = scores[:, num_cls:]
                pred_cls_idx = torch.argmax(positive_scores, dim=1)
                pos_score = positive_scores.gather(1, pred_cls_idx.unsqueeze(1)).squeeze(1)
                neg_score = negative_scores.gather(1, pred_cls_idx.unsqueeze(1)).squeeze(1)
            margins = pos_score - neg_score

            # 类内区分边际: top1 - mean（而非 top1-top2）
            # 已知类: top1 >> mean → cls_margin大  未知类: 各pos接近 → cls_margin小
            top1 = torch.topk(positive_scores, k=1, dim=1).values[:, 0]
            mean_pos = positive_scores.mean(dim=1)
            cls_margins = top1 - mean_pos

            # Session-aware 阈值平滑（默认 blend=0 即透明回退自适应阈值）
            blend = float(getattr(args, 'session_thr_blend', 0.0))
            thr_open, thr_cls = _session_aware_threshold(margins, cls_margins, session, blend=blend)
            # Task 1.6: OSR 阈值缩放（<1 更宽容 → 更多样本保留为 known）
            osr_scale = float(getattr(args, 'osr_thr_scale', 1.0))
            if osr_scale != 1.0:
                thr_open = thr_open * osr_scale
                thr_cls = thr_cls * osr_scale
            # OSR 判定: 是否同时使用 cls_margin 作为第二约束
            use_cls_margin = bool(getattr(args, 'osr_use_cls_margin', False))
            if use_cls_margin:
                unknown_mask = (margins <= thr_open) & (cls_margins <= thr_cls)
            else:
                unknown_mask = margins <= thr_open
            known_mask = ~unknown_mask

            # ===== Per-sample OSR 诊断 =====
            print(f'\n  [PER-SAMPLE] session={session} thr_open={thr_open:.4f} thr_cls={thr_cls:.4f} '
                  f'avg_margin={margins.mean().item():.4f} avg_clsmargin={cls_margins.mean().item():.4f}')
            _nl = args.num_labeled_classes
            for j in range(label.size(0)):
                _lbl = label[j].item()
                _is_known = _lbl < _nl
                _pred = 'K' if bool(known_mask[j].item()) else 'U'
                _correct = 'Y' if (_is_known and _pred == 'K') or (not _is_known and _pred == 'U') else 'N'
                print(f'    [{_correct}] cls={_lbl:3d} gt={"K" if _is_known else "U"} pred={_pred} '
                      f'margin={margins[j].item():+.4f} pos={pos_score[j].item():.4f} '
                      f'neg={neg_score[j].item():.4f} clsmargin={cls_margins[j].item():.4f}')
            # ===== Per-sample 诊断结束 =====

            # Accumulate raw scores; compute one session-level AUROC after all
            # batches instead of averaging batch AUROCs.
            _nl = args.num_labeled_classes
            session_labels.extend(label.detach().cpu().view(-1).tolist())
            session_known_margins.extend(margins.detach().cpu().view(-1).tolist())
            for j in range(label.size(0)):
                ranked_novel_candidates.append(
                    (float(margins[j].detach().cpu()), data[j].view(1, -1), label[j].item()))

            for j in range(label.size(0)):
                if bool(known_mask[j].item()):
                    knowns.append(data[j].view(1, -1))
                    klabels.append(label[j].item())
                    accepted_boundary_distance.append(float((margins[j] - thr_open).detach().cpu()))
                else:
                    unknowns.append(data[j].view(1, -1))
                    unlabels.append(label[j].item())
    
    osr_metrics = _unknown_metrics(session_labels, session_known_margins,
                                   args.num_labeled_classes)
    session_auroc = osr_metrics['auroc']
    if not hasattr(run_test_fsl, '_auroc_list'):
        run_test_fsl._auroc_list = []
    if np.isfinite(session_auroc):
        run_test_fsl._auroc_list.append(session_auroc)
    if not hasattr(run_test_fsl, '_osr_metrics_list'):
        run_test_fsl._osr_metrics_list = []
    run_test_fsl._osr_metrics_list.append(osr_metrics)
    print(f'  [OSR-METRIC] session={session} AUROC={osr_metrics["auroc"]:.4f} '
          f'AUPR={osr_metrics["aupr"]:.4f} FPR95={osr_metrics["fpr95"]:.4f}')
    # Side-channel retained for backward-compatible callers. Distances are
    # aligned with ``knowns`` and are small for accepted samples nearest the
    # known/unknown decision boundary.
    run_test_fsl._last_accepted_boundary_distance = accepted_boundary_distance
    ranked_novel_candidates.sort(key=lambda item: item[0])
    run_test_fsl._last_ranked_novel_data = [item[1] for item in ranked_novel_candidates]
    # Labels are diagnostic only; neither ranking nor clustering reads them.
    run_test_fsl._last_ranked_novel_labels = [item[2] for item in ranked_novel_candidates]
    return unknowns, unlabels, knowns, klabels
def plot(args,model,test_fsl_loader):
        model.eval()
        result1,result2,thr1,thr2 = [],[],[],[]
        proto = model.fc.weight[:args.num_labeled_classes,:].detach()
        for idx,batch in enumerate(test_fsl_loader,1):
            data, label = [_.cuda() for _ in batch]
            data,label = data.squeeze(),label.squeeze()   
            feature = model.encode(data)
            all_prob= compute_feats(model, label[:args.n_ways*5],feature,proto)

            # 绘制热力图
            # query_probs = all_prob
            # similarity_matrix = query_probs.cpu().detach().numpy()
            # plt.figure(figsize=(12, 8))
            # sns.heatmap(similarity_matrix, cmap='viridis', cbar=True)
            # plt.title("Similarity Matrix Heatmap")
            # plt.xlabel("Categories")
            # plt.ylabel("Samples")
            # plt.show()
            # plt.savefig('/data/jessy/open world/new_save'+'/Heatmap.png')

            thr = all_prob[:,-1]
            query,_ = torch.max(all_prob[:,:args.num_labeled_classes],dim=1)
            result1.append(query[:args.n_ways*args.n_shots])
            result2.append(query[args.n_ways*args.n_shots:])
            thr1.append(thr[:args.n_ways*args.n_shots])
            thr2.append(thr[args.n_ways*args.n_shots:])
            
        result1 = torch.cat(result1)
        result2 = torch.cat(result2)
        thr1 = torch.cat(thr1)
        thr2 = torch.cat(thr2)
        count, bins, ignored = plt.hist(result1.tolist(), bins=30, alpha=0.75, color='darkorange')
        plt.hist(result2.tolist(), bins=30, alpha=0.75, color='lightgreen')
        # plt.hist(thr1.tolist(), bins=30, alpha=0.75, color='bisque')
        # plt.hist(thr2.tolist(), bins=30, alpha=0.75, color='yellow')
        data_sorted = sorted(zip(result1.tolist(), thr1.tolist()), key=lambda x: x[0])
        data_sorted, thresholds_sorted = zip(*data_sorted)
        # 绘制点图以展示阈值比较
        #plt.scatter(data_sorted, thresholds_sorted, color='red', label='Thresholds', alpha=0.5)
        #plt.plot(bin_centers, line_y_values, 'r--', linewidth=2)
        plt.xlabel('score')
        plt.ylabel('Frequency')
        plt.show()
        output_dir = str(getattr(args, 'save_result', 'outputs'))
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'hist.png'))

def compute_feats(model, label_id, features, proto, use_uop=False, uop_chunk_size=5, uop_mode='antimean',
                   osr_noise_std=0.1):
    with torch.no_grad():
        test_cosine_scores = model.cls_classifier.incre_forward(
            features, proto, label_id,
            use_uop=use_uop, uop_chunk_size=uop_chunk_size, uop_mode=uop_mode,
            osr_noise_std=osr_noise_std,
        )
        # 1对1开集判决使用 raw logits 更稳定，避免 softmax 的跨类耦合干扰
        query_cls_scores = test_cosine_scores.detach().squeeze()
        if query_cls_scores.dim() == 1:
            query_cls_scores = query_cls_scores.unsqueeze(0)
    return query_cls_scores


def known_test(args,model,data,label):
    feats=[]
    label = torch.tensor(label)
    model = model.eval()
    for i in range(len(data)):
        feat = model.encode(data[i])
        feats.append(feat)
    proto = model.fc.weight[:args.num_labeled_classes,:].detach().unsqueeze(0).unsqueeze(0)
    feats = torch.stack(feats)
    logits=F.cosine_similarity(feats, proto, dim=-1)
    logits=torch.squeeze(logits)
    acc = count_acc(logits, label.to('cuda'))
    preds = torch.argmax(logits, dim=1)
    score = f1_score(label.cpu().numpy(),preds.cpu().numpy(),average='macro')
    return acc,score


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h       
        
