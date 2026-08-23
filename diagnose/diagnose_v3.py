"""
系统诊断脚本 (只读)
用途：用当前 save/epoch_15.pth 一次性产出：
  - 原型几何指标 (base / new / 对偶原型)
  - 特征空间一致性 (encode vs hgnn_encode)
  - 规模失配诊断 (AttnClassifier.calibrator 在 5/80/100 way 下的 attention 熵)
  - Session-level AUROC / OSCR / FPR@95 (补齐指标)
  - 4 张关键图 + metrics.csv + report.md
不改任何现有代码，完全旁路式分析。
"""
import os
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
sys.path.insert(0, '/data/lqq/baseline')

import argparse
import copy
import csv
import json
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# 复用项目既有模块
from network import MYNET, replace_base_fc
from train_unopenset import (
    args_parser, dict2namespace, set_up_datasets, set_seed,
    debug_cluster, test as test_cosine, known_test,
)
from threshold_free import (
    compute_feats, _adaptive_margin_threshold, reset_session_stats,
)
from data.dataloader import (
    get_pretrain_dataloader, get_dataloader, get_testloader, get_inc_testloader,
)
from enhance_module import LocalFeatureCluster

_LFC_CACHE = None
def _feat_pipeline(model, data, device):
    """严格复刻 run_test_fsl 中的特征流：hgnn_encode + LocalFeatureCluster(0.3) if enabled."""
    global _LFC_CACHE
    with torch.no_grad():
        feat = model.hgnn_encode(data)
        if hasattr(model, 'feature_enhance'):
            if _LFC_CACHE is None:
                _LFC_CACHE = LocalFeatureCluster(k_ratio=0.3).to(device)
            feat, _ = _LFC_CACHE(feat)
        feat = feat.to(device)
    return feat

OUT_DIR = '/data/lqq/baseline/save_result/diagnose_v3'
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================================================
# 工具
# ==========================================================================
def _log(msg):
    print(f"[DIAG] {msg}", flush=True)

def _cos(a, b):
    """pairwise cosine similarity, returns [A.size(0), B.size(0)]"""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()

def _entropy(p, eps=1e-12):
    """row-wise entropy of a probability matrix"""
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


# ==========================================================================
# 参数组装：复用 train_unopenset 的 parser，但用 fast_eval 配置
# ==========================================================================
def build_args():
    parser = argparse.ArgumentParser('diag', parents=[args_parser()])
    raw = parser.parse_args([
        '-config', '/data/lqq/baseline/configs/fast_eval.yml',
        '--checkpoint', 'True',
        '--checkpoint_name', 'epoch_15.pth',
        '--opt_version', 'diagnose_v3',
        '--run_tag', 'diag',
    ])
    with open(raw.config) as f:
        cfg = yaml.safe_load(f)['train']
    cfg.update(vars(raw))
    args = dict2namespace(cfg)
    args.cuda = torch.cuda.is_available()
    return args


# ==========================================================================
# 载入模型：复制 train_unopenset.train() 的 checkpoint 逻辑
# ==========================================================================
def load_model(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    model = MYNET(args, mode='encoder').to(device)
    ckpt_path = os.path.join(args.save_dir, args.checkpoint_name)
    _log(f"loading checkpoint from {ckpt_path}")
    params = torch.load(ckpt_path, weights_only=True)['cls_params']
    cls_params = {k: v for k, v in params.items() if 'fc' in k}
    model.cls_classifier.init_representation(cls_params)
    sd = model.state_dict()
    sd.update(params)
    model.load_state_dict(sd)
    model.eval()
    return model, device


# ==========================================================================
# Phase A: 原型几何指标
# ==========================================================================
def phase_a_prototype_geometry(model, args, device):
    _log("== Phase A: 原型几何 ==")
    rec = OrderedDict()

    # meta_train 结束时的 fc.weight (在 replace_base_fc 前)
    fc_meta = model.fc.weight.data.clone().to(device)       # [100, 512]
    # AttnClassifier 内部冻结的 base_weight
    weight_base = model.cls_classifier.weight_base.data.clone().to(device)  # [100, 512]

    # base 类的 meta 原型 vs replace 均值原型（稍后 replace 完再填）
    rec['fc_meta_base_pairwise_cos_mean'] = float(_cos(fc_meta[:80], fc_meta[:80]).fill_diagonal_(0).sum() / (80*79))
    rec['fc_meta_base_pairwise_cos_min'] = float(_cos(fc_meta[:80], fc_meta[:80]).fill_diagonal_(2).min())
    rec['fc_meta_base_pairwise_cos_max'] = float(_cos(fc_meta[:80], fc_meta[:80]).fill_diagonal_(-2).max())

    # replace_base_fc：用训练集均值覆盖 fc.weight[:80]
    trainset, _ = get_pretrain_dataloader(args)
    model = replace_base_fc(args, trainset, model)
    fc_after = model.fc.weight.data.clone().to(device)

    rec['fc_replace_base_pairwise_cos_mean'] = float(_cos(fc_after[:80], fc_after[:80]).fill_diagonal_(0).sum() / (80*79))
    rec['fc_replace_base_pairwise_cos_min'] = float(_cos(fc_after[:80], fc_after[:80]).fill_diagonal_(2).min())
    rec['fc_replace_base_pairwise_cos_max'] = float(_cos(fc_after[:80], fc_after[:80]).fill_diagonal_(-2).max())

    # meta 与 replace 均值原型的对齐度（每类一对一 cos）
    diag_cos = (F.normalize(fc_meta[:80], dim=-1) * F.normalize(fc_after[:80], dim=-1)).sum(-1)
    rec['fc_meta_vs_replace_diag_cos_mean'] = float(diag_cos.mean())
    rec['fc_meta_vs_replace_diag_cos_min'] = float(diag_cos.min())

    # 对偶原型：用 fc.weight[:80] 作为 proto 跑 cls_classifier.calibrator + open_generator
    with torch.no_grad():
        proto = fc_after[:80].to(device)
        dummy_cls_ids = torch.arange(80, device=device)
        # SupportCalibrator 期望 [B, N, D]; 这里直接按 incre_forward 逻辑走
        supp_protos = model.cls_classifier.calibrator(proto, weight_base[dummy_cls_ids], True)
        # supp_protos: [80, 1, 512] 或 [80, 512]
        supp_protos = supp_protos.view(-1, supp_protos.size(-1))
        fakeclass_protos, _ = model.cls_classifier.open_generator(supp_protos, weight_base[dummy_cls_ids], True)
        fakeclass_protos = fakeclass_protos.view(-1, fakeclass_protos.size(-1))

    # pos vs neg 一一对应的 cos
    diag_pn = (F.normalize(supp_protos, dim=-1) * F.normalize(fakeclass_protos, dim=-1)).sum(-1)
    rec['pos_vs_neg_cos_mean'] = float(diag_pn.mean())
    rec['pos_vs_neg_cos_min'] = float(diag_pn.min())
    rec['pos_vs_neg_cos_max'] = float(diag_pn.max())

    # neg 互相之间多样性
    neg_pair = _cos(fakeclass_protos, fakeclass_protos).fill_diagonal_(0)
    rec['neg_pairwise_cos_mean'] = float(neg_pair.sum() / (80*79))
    rec['neg_pairwise_cos_max'] = float(neg_pair.max())

    # supp 和 fakeclass 的均值向量距离（整体分布中心是否分离）
    rec['pos_center_vs_neg_center_cos'] = float(
        (F.normalize(supp_protos.mean(0, keepdim=True), dim=-1) *
         F.normalize(fakeclass_protos.mean(0, keepdim=True), dim=-1)).sum().item()
    )

    _log(f"  fc_meta_base_pairwise_cos_mean = {rec['fc_meta_base_pairwise_cos_mean']:.4f}")
    _log(f"  fc_replace_base_pairwise_cos_mean = {rec['fc_replace_base_pairwise_cos_mean']:.4f}")
    _log(f"  fc_meta_vs_replace_diag_cos_mean = {rec['fc_meta_vs_replace_diag_cos_mean']:.4f}")
    _log(f"  pos_vs_neg_cos_mean = {rec['pos_vs_neg_cos_mean']:.4f}")
    _log(f"  neg_pairwise_cos_mean = {rec['neg_pairwise_cos_mean']:.4f}")

    return rec, model, fc_meta, fc_after, supp_protos.detach(), fakeclass_protos.detach()


# ==========================================================================
# Phase B: 特征空间一致性
# ==========================================================================
def phase_b_feature_space_consistency(model, args, device):
    _log("== Phase B: 特征空间一致性 ==")
    rec = OrderedDict()

    # 取 base session 测试集的一个 batch
    _, loader = get_testloader(args, 0)
    model.eval()
    model.mode = 'incre'
    batch = next(iter(loader))
    data = batch[0].to(device)[:64]

    with torch.no_grad():
        feat_encode = model.encode(data)                  # [B, 512]
        if feat_encode.dim() > 2:
            feat_encode = feat_encode.view(feat_encode.size(0), -1)
        feat_hgnn = model.hgnn_encode(data)               # 可能是 [B, 512, H, W]
        if feat_hgnn.dim() == 4:
            feat_hgnn = feat_hgnn.mean(dim=[2, 3])

    cos_paired = (F.normalize(feat_encode, dim=-1) * F.normalize(feat_hgnn, dim=-1)).sum(-1)
    rec['encode_vs_hgnn_paired_cos_mean'] = float(cos_paired.mean())
    rec['encode_vs_hgnn_paired_cos_min'] = float(cos_paired.min())
    rec['encode_vs_hgnn_paired_cos_std'] = float(cos_paired.std())

    _log(f"  encode_vs_hgnn_paired_cos_mean = {rec['encode_vs_hgnn_paired_cos_mean']:.4f} "
         f"(min={rec['encode_vs_hgnn_paired_cos_min']:.4f})")
    return rec


# ==========================================================================
# Phase C: 规模失配诊断
# ==========================================================================
def phase_c_scale_mismatch(model, fc_after, weight_base, device):
    _log("== Phase C: 规模失配诊断 (calibrator attention 熵) ==")
    rec = OrderedDict()

    # 手动复现 calibrator 的内部 attention: q=proto, k=proto, v=proto (neg_gen_type='att')
    # 在 SupportCalibrator(neg_gen_type='att') 下 base_weights = support_feat
    # 故真实 attention score 是 proto @ proto.T / sqrt(d_k)
    calib = model.cls_classifier.calibrator
    mha = calib.calibrator  # MultiHeadAttention
    d_k = mha.d_k
    temperature = float(np.sqrt(d_k))

    def attn_entropy(proto):
        """复现 MultiHeadAttention 的 attention 计算，用投影后的 q/k 计算 attn entropy。"""
        with torch.no_grad():
            # proto: [N, D] -> [N, 1, D] (SupportCalibrator view 逻辑)
            sup = proto.view(-1, 1, proto.size(-1))
            # 对于 neg_gen_type='att', k=v=sup (论文中的 self-attention)
            # MultiHeadAttention.forward: q/k/v 投影 -> attn_score = bmm(q,k^T)/temperature -> softmax
            n_head = mha.n_head
            q = mha.w_qs(sup).view(-1, 1, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, 1, d_k)
            # k 需要来自所有其它 proto（而不是 sup 的 1 维）
            k_all = mha.w_ks(proto.unsqueeze(0).expand(sup.size(0), -1, -1)).view(
                sup.size(0), proto.size(0), n_head, d_k
            ).permute(2, 0, 1, 3).contiguous().view(-1, proto.size(0), d_k)
            # attn: [n_head*N, 1, N]
            attn = torch.bmm(q, k_all.transpose(1, 2)) / temperature
            attn = F.softmax(attn, dim=-1).squeeze(1)  # [N, N]
            return _entropy(attn).mean().item(), attn

    # 模拟 3 种规模：5-way (训练分布)、80-way (session 0)、100-way (session 4 末)
    # 5-way: 随机选 5 个 base proto
    proto_80 = fc_after[:80].to(device)
    proto_100 = fc_after[:100].to(device)

    rng = np.random.RandomState(42)
    idx_5 = rng.choice(80, 5, replace=False)
    proto_5 = proto_80[torch.tensor(idx_5, device=device)]

    ent5, attn5 = attn_entropy(proto_5)
    ent80, attn80 = attn_entropy(proto_80)
    ent100, attn100 = attn_entropy(proto_100)

    max_ent_5 = float(np.log(5))
    max_ent_80 = float(np.log(80))
    max_ent_100 = float(np.log(100))

    rec['attn_entropy_5way'] = ent5
    rec['attn_entropy_80way'] = ent80
    rec['attn_entropy_100way'] = ent100
    rec['attn_entropy_5way_normalized'] = ent5 / max_ent_5   # 0=一点集中, 1=完全平均
    rec['attn_entropy_80way_normalized'] = ent80 / max_ent_80
    rec['attn_entropy_100way_normalized'] = ent100 / max_ent_100
    rec['attn_top1_prob_5way'] = float(attn5.max(-1).values.mean())
    rec['attn_top1_prob_80way'] = float(attn80.max(-1).values.mean())
    rec['attn_top1_prob_100way'] = float(attn100.max(-1).values.mean())

    _log(f"  5-way  entropy={ent5:.3f}/{max_ent_5:.3f} (norm {rec['attn_entropy_5way_normalized']:.3f}), "
         f"top1 prob={rec['attn_top1_prob_5way']:.3f}")
    _log(f"  80-way entropy={ent80:.3f}/{max_ent_80:.3f} (norm {rec['attn_entropy_80way_normalized']:.3f}), "
         f"top1 prob={rec['attn_top1_prob_80way']:.3f}")
    _log(f"  100-way entropy={ent100:.3f}/{max_ent_100:.3f} (norm {rec['attn_entropy_100way_normalized']:.3f}), "
         f"top1 prob={rec['attn_top1_prob_100way']:.3f}")

    return rec, attn80


# ==========================================================================
# Phase D: Session 级 OSR 诊断 (AUROC / OSCR / FPR@95)
# ==========================================================================
def diag_run_session_osr(model, args, loader, session, device):
    """复制 run_test_fsl 的前半段，同时保留真值 label 以便算 AUROC/OSCR。
    session s 下，真实 unknown = (label >= num_base + (s-1)*5)
    """
    model.eval()
    proto = model.fc.weight[:args.num_labeled_classes, :].detach()
    n_ways = args.n_ways
    num_cls = args.num_labeled_classes

    all_margins = []
    all_cls_margins = []
    all_pos_scores = []
    all_pred_cls_idx = []
    all_is_truly_unknown = []
    all_labels = []
    all_pred_top1_correct = []      # closed-set correctness among known
    known_unknown_boundary = args.num_base + (session - 1) * args.way if session >= 1 else args.num_base

    with torch.no_grad():
        for batch in loader:
            data, label = [x.cuda() for x in batch]
            data, label = data.squeeze(), label.squeeze()
            feat = _feat_pipeline(model, data, device)
            scores = compute_feats(model, label[:n_ways*5], feat, proto)
            pos_scores = scores[:, :num_cls]
            neg_scores = scores[:, num_cls:]

            pred_cls_idx = torch.argmax(pos_scores, dim=1)
            pos = pos_scores.gather(1, pred_cls_idx.unsqueeze(1)).squeeze(1)
            neg = neg_scores.gather(1, pred_cls_idx.unsqueeze(1)).squeeze(1)
            margins = pos - neg

            top2 = torch.topk(pos_scores, k=min(2, pos_scores.size(1)), dim=1).values
            cls_margins = top2[:, 0] - top2[:, 1] if top2.size(1) > 1 else top2[:, 0]

            truly_unknown = (label >= known_unknown_boundary)
            all_margins.append(margins.cpu())
            all_cls_margins.append(cls_margins.cpu())
            all_pos_scores.append(pos.cpu())
            all_pred_cls_idx.append(pred_cls_idx.cpu())
            all_is_truly_unknown.append(truly_unknown.cpu())
            all_labels.append(label.cpu())
            # closed-set correctness: pred_cls_idx == label (对于 known 样本)
            all_pred_top1_correct.append((pred_cls_idx.cpu() == label.cpu()))

    margins = torch.cat(all_margins).numpy()
    cls_margins = torch.cat(all_cls_margins).numpy()
    pos_scores_np = torch.cat(all_pos_scores).numpy()
    is_unk_np = torch.cat(all_is_truly_unknown).numpy().astype(int)
    labels_np = torch.cat(all_labels).numpy()
    correct_np = torch.cat(all_pred_top1_correct).numpy().astype(int)

    out = {
        'margins': margins, 'cls_margins': cls_margins, 'pos': pos_scores_np,
        'is_unknown': is_unk_np, 'labels': labels_np, 'correct_topk': correct_np,
    }

    # AUROC: margin 越大越 known, 所以 score = -margin 让 unknown 排在高分
    try:
        out['auroc_margin'] = float(roc_auc_score(is_unk_np, -margins))
    except Exception:
        out['auroc_margin'] = float('nan')
    # AUROC: pos_score 越大越 known, 同理取反
    try:
        out['auroc_posscore'] = float(roc_auc_score(is_unk_np, -pos_scores_np))
    except Exception:
        out['auroc_posscore'] = float('nan')

    # FPR@TPR=0.95 (TPR 定义为 unknown 被拒绝率)
    try:
        fpr, tpr, _ = roc_curve(is_unk_np, -margins)
        idx = np.argmin(np.abs(tpr - 0.95))
        out['fpr_at_tpr95'] = float(fpr[idx])
    except Exception:
        out['fpr_at_tpr95'] = float('nan')

    # OSCR: 扫描 threshold 下 CCR (已知且分类正确) vs FPR (未知被误判为已知)
    out['oscr'] = compute_oscr(-margins, correct_np, is_unk_np)
    return out


def compute_oscr(scores, correct, is_unk):
    """
    scores: 越大越未知 (unknown-score)
    correct: 闭集分类是否正确 (仅对 known 有意义)
    is_unk: 1=真未知, 0=真已知
    CCR(t) = P(score<=t AND correct | known)
    FPR(t) = P(score<=t | unknown)  (未知被判为已知的比例)
    OSCR = AUC of CCR vs FPR
    """
    known_mask = is_unk == 0
    unk_mask = is_unk == 1
    if known_mask.sum() == 0 or unk_mask.sum() == 0:
        return float('nan')
    scores_k = scores[known_mask]
    correct_k = correct[known_mask]
    scores_u = scores[unk_mask]

    # 把所有 threshold 作为候选
    ts = np.sort(np.unique(np.concatenate([scores_k, scores_u])))
    ts = np.concatenate([[ts[0] - 1e-6], ts, [ts[-1] + 1e-6]])
    ccr_list, fpr_list = [], []
    for t in ts:
        known_accept = scores_k <= t
        ccr = float((known_accept & (correct_k == 1)).sum() / max(1, known_mask.sum()))
        fpr = float((scores_u <= t).sum() / max(1, unk_mask.sum()))
        ccr_list.append(ccr)
        fpr_list.append(fpr)
    # OSCR = 按 FPR 排序的 trapz
    order = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[order]
    ccr_sorted = np.array(ccr_list)[order]
    return float(np.trapz(ccr_sorted, fpr_sorted))


def phase_d_session_diagnostics(model, args, device):
    _log("== Phase D: Session-level OSR 诊断 ==")
    # 只跑 1 轮 test_time 即可 (诊断不需要平均)
    args.current_test = 0
    args.num_labeled_classes = args.num_base
    reset_session_stats()

    # Session 0 baseline
    per_session = []
    _, base_testloader = get_testloader(args, 0)
    base_acc = test_cosine(args, model, base_testloader, 0)
    per_session.append({
        'session': 0, 'known_acc': base_acc, 'unknown_acc': 0.0,
        'inc_acc': 0.0, 'all_acc': base_acc,
        'auroc_margin': float('nan'), 'auroc_posscore': float('nan'),
        'oscr': float('nan'), 'fpr_at_tpr95': float('nan'),
    })

    session_raw = {}  # 保存原始 margin/label 以供画图
    for session in range(args.start_session, args.num_session):
        _log(f"  -- session {session} --")
        model.mode = args.network.new_mode
        model.eval()
        _, unlabelled_loader = get_dataloader(args, session)

        # OSR + 诊断数据
        osr_diag = diag_run_session_osr(model, args, unlabelled_loader, session, device)
        session_raw[session] = osr_diag

        # 为了后续 debug_cluster & known_test，复刻 threshold_free 的判决流程
        margins_t = torch.tensor(osr_diag['margins'])
        cls_margins_t = torch.tensor(osr_diag['cls_margins'])
        thr_open = _adaptive_margin_threshold(margins_t)
        thr_cls = float(torch.quantile(cls_margins_t, 0.35).item())
        unknown_mask = (margins_t <= thr_open) & (cls_margins_t <= thr_cls)
        osr_diag['thr_open'] = thr_open
        osr_diag['thr_cls'] = thr_cls
        osr_diag['pred_unknown'] = unknown_mask.numpy().astype(int)

        # 实际分 known/unknown 数据做 debug_cluster + known_test
        # 注意：需要重新遍历 dataloader (已被 diag_run_session_osr 消耗过，不能直接用)
        knowns, klabels, unknowns, ulabels = [], [], [], []
        proto = model.fc.weight[:args.num_labeled_classes, :].detach()
        idx_ptr = 0
        for batch in unlabelled_loader:
            data, label = [x.cuda() for x in batch]
            data, label = data.squeeze(), label.squeeze()
            feat = _feat_pipeline(model, data, device)
            with torch.no_grad():
                scores = compute_feats(model, label[:args.n_ways*5], feat, proto)
            num_cls = args.num_labeled_classes
            pred_cls_idx = torch.argmax(scores[:, :num_cls], dim=1)
            pos = scores[:, :num_cls].gather(1, pred_cls_idx.unsqueeze(1)).squeeze(1)
            neg = scores[:, num_cls:].gather(1, pred_cls_idx.unsqueeze(1)).squeeze(1)
            m = (pos - neg).cpu()
            top2 = torch.topk(scores[:, :num_cls], k=min(2, num_cls), dim=1).values
            cm = (top2[:, 0] - top2[:, 1] if top2.size(1) > 1 else top2[:, 0]).cpu()
            batch_unk = (m <= thr_open) & (cm <= thr_cls)
            for j in range(label.size(0)):
                if bool(batch_unk[j].item()):
                    unknowns.append(data[j].view(1, -1))
                    ulabels.append(label[j].item())
                else:
                    knowns.append(data[j].view(1, -1))
                    klabels.append(label[j].item())

        # debug_cluster 会改写 model.fc.weight[80+]，但诊断后会重新 replace/覆盖所以 OK
        cluster_acc = debug_cluster(args, model, unknowns, ulabels, session)
        acc_known, _ = known_test(args, model, knowns, klabels)

        # incremental / all
        _, testloader = get_testloader(args, session)
        all_acc = test_cosine(args, model, testloader, session)
        _, inc_testloader = get_inc_testloader(args, session)
        inc_acc = test_cosine(args, model, inc_testloader, session)

        per_session.append({
            'session': session,
            'known_acc': acc_known,
            'unknown_acc': cluster_acc,
            'inc_acc': inc_acc,
            'all_acc': all_acc,
            'auroc_margin': osr_diag['auroc_margin'],
            'auroc_posscore': osr_diag['auroc_posscore'],
            'oscr': osr_diag['oscr'],
            'fpr_at_tpr95': osr_diag['fpr_at_tpr95'],
            'num_known_pred': int((~unknown_mask).sum()),
            'num_unknown_pred': int(unknown_mask.sum()),
            'num_truly_unknown': int(osr_diag['is_unknown'].sum()),
            'num_truly_known': int((osr_diag['is_unknown'] == 0).sum()),
            'thr_open': thr_open,
            'thr_cls': thr_cls,
        })
        args.num_labeled_classes += args.way

    return per_session, session_raw


# ==========================================================================
# 持久化 + 主入口
# ==========================================================================
def write_csv(rows, path):
    if not rows:
        return
    # union of all keys across rows to avoid DictWriter fieldnames mismatch
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    t0 = time.time()
    args = build_args()
    set_up_datasets(args)
    set_seed(args.seed)

    model, device = load_model(args)

    # Phase A + replace_base_fc
    rec_a, model, fc_meta, fc_after, supp_protos, fakeclass_protos = \
        phase_a_prototype_geometry(model, args, device)

    # Phase B
    rec_b = phase_b_feature_space_consistency(model, args, device)

    # Phase C
    weight_base = model.cls_classifier.weight_base.data.to(device)
    rec_c, attn80 = phase_c_scale_mismatch(model, fc_after, weight_base, device)

    # Phase D
    session_rows, session_raw = phase_d_session_diagnostics(model, args, device)

    # 存盘：汇总
    overall = OrderedDict()
    overall.update(rec_a)
    overall.update(rec_b)
    overall.update(rec_c)
    # 汇总 session 指标
    for k in ['known_acc', 'unknown_acc', 'inc_acc', 'all_acc',
              'auroc_margin', 'auroc_posscore', 'oscr', 'fpr_at_tpr95']:
        vals = [r[k] for r in session_rows if r['session'] > 0]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            overall[f'avg_{k}'] = float(np.mean(vals))

    # 额外：保存 tensor 供可视化
    np.save(os.path.join(OUT_DIR, 'supp_protos.npy'), supp_protos.cpu().numpy())
    np.save(os.path.join(OUT_DIR, 'fakeclass_protos.npy'), fakeclass_protos.cpu().numpy())
    np.save(os.path.join(OUT_DIR, 'fc_meta.npy'), fc_meta.cpu().numpy())
    np.save(os.path.join(OUT_DIR, 'fc_after.npy'), fc_after.cpu().numpy())
    np.save(os.path.join(OUT_DIR, 'attn80.npy'), attn80.cpu().numpy())
    for s, d in session_raw.items():
        np.savez(os.path.join(OUT_DIR, f'session_{s}_raw.npz'),
                 margins=d['margins'], cls_margins=d['cls_margins'],
                 is_unknown=d['is_unknown'], pos=d['pos'], labels=d['labels'],
                 correct=d['correct_topk'])

    write_csv(session_rows, os.path.join(OUT_DIR, 'sessions.csv'))
    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'overall': overall, 'sessions': session_rows}, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)

    _log(f"done, elapsed {time.time()-t0:.1f}s")
    _log("overall metrics:")
    for k, v in overall.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
