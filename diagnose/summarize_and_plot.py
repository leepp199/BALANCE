"""
从 save_result/diagnose_v3 下已落盘的 npz/npy 汇总并生成:
- metrics.csv: session-level AUROC/OSCR/FPR@TPR95/inc/all
- plot_A_proto_tsne.png: base_proto + new_proto + fakeclass t-SNE
- plot_B_margin_scatter.png: 每 session 的 margin x cls_margin 散点
- plot_C_attention80.png: 80-way calibrator attention 热力图
- plot_D_proto_geometry.png: 原型几何诊断三联图
- report.md: 关键结论
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import TSNE

D = '/data/lqq/baseline/save_result/diagnose_v3'
OUT_CSV = os.path.join(D, 'metrics.csv')
OUT_REPORT = os.path.join(D, 'report.md')

# ----------------------- 加载 raw -----------------------
fc_meta = np.load(os.path.join(D, 'fc_meta.npy'))          # (100,512)
fc_after = np.load(os.path.join(D, 'fc_after.npy'))        # (100,512)
supp = np.load(os.path.join(D, 'supp_protos.npy'))         # (80,512) pos
fake = np.load(os.path.join(D, 'fakeclass_protos.npy'))    # (80,512) neg
attn80 = np.load(os.path.join(D, 'attn80.npy'))            # (80,80)

NUM_BASE = 80


# ----------------------- 工具函数 -----------------------
def cos_sim(a, b):
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return an @ bn.T


def pairwise_offdiag_mean(M):
    N = M.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return float(M[mask].mean())


def compute_oscr(scores, correct, is_unk):
    known_mask = is_unk == 0
    unk_mask = is_unk == 1
    if known_mask.sum() == 0 or unk_mask.sum() == 0:
        return float('nan')
    scores_k = scores[known_mask]
    correct_k = correct[known_mask]
    scores_u = scores[unk_mask]
    ts = np.sort(np.unique(np.concatenate([scores_k, scores_u])))
    ts = np.concatenate([[ts[0] - 1e-6], ts, [ts[-1] + 1e-6]])
    ccr_list, fpr_list = [], []
    for t in ts:
        ccr = float(((scores_k <= t) & (correct_k == 1)).sum() / max(1, known_mask.sum()))
        fpr = float((scores_u <= t).sum() / max(1, unk_mask.sum()))
        ccr_list.append(ccr); fpr_list.append(fpr)
    order = np.argsort(fpr_list)
    return float(np.trapz(np.array(ccr_list)[order], np.array(fpr_list)[order]))


# ----------------------- Phase A: 原型几何 -----------------------
base_meta = fc_meta[:NUM_BASE]
base_after = fc_after[:NUM_BASE]
new_after = fc_after[NUM_BASE:]  # (20,512)

sim_meta_base = cos_sim(base_meta, base_meta)
sim_after_base = cos_sim(base_after, base_after)
# 同类 meta vs after 对角
diag_ma = np.array([
    np.dot(base_meta[i], base_after[i]) /
    (np.linalg.norm(base_meta[i]) * np.linalg.norm(base_after[i]) + 1e-8)
    for i in range(NUM_BASE)
])
# new 原型与 base 最小距离
sim_new2base = cos_sim(new_after, base_after)  # (20,80)
min_sim_new2base = sim_new2base.max(axis=1)    # 每个 new 最近 base 的相似度

# 对偶原型
pos_vs_neg = np.array([
    np.dot(supp[i], fake[i]) / (np.linalg.norm(supp[i]) * np.linalg.norm(fake[i]) + 1e-8)
    for i in range(NUM_BASE)
])
neg_pair = cos_sim(fake, fake)

geom = {
    'fc_meta_base_pair_cos_mean': pairwise_offdiag_mean(sim_meta_base),
    'fc_meta_base_pair_cos_max':  float(sim_meta_base[~np.eye(NUM_BASE, dtype=bool)].max()),
    'fc_after_base_pair_cos_mean': pairwise_offdiag_mean(sim_after_base),
    'fc_after_base_pair_cos_max':  float(sim_after_base[~np.eye(NUM_BASE, dtype=bool)].max()),
    'meta_vs_after_same_class_cos_mean': float(diag_ma.mean()),
    'meta_vs_after_same_class_cos_min':  float(diag_ma.min()),
    'new_to_nearest_base_cos_mean': float(min_sim_new2base.mean()),
    'new_to_nearest_base_cos_max':  float(min_sim_new2base.max()),
    'pos_vs_neg_cos_mean': float(pos_vs_neg.mean()),
    'pos_vs_neg_cos_min':  float(pos_vs_neg.min()),
    'neg_pair_cos_mean':   pairwise_offdiag_mean(neg_pair),
}

# ----------------------- Phase C: attention entropy (已从 log 读) -----------------------
# 从 run.log 直接取值;若要重算可用 attn80
# attn80 行和应近似 1
ent80 = -(attn80 * np.log(np.clip(attn80, 1e-12, 1))).sum(axis=1).mean()
ent80_norm = float(ent80 / np.log(NUM_BASE))
top1_80 = float(attn80.max(axis=1).mean())

# ----------------------- Session-level metrics -----------------------
session_rows = []
session_rows.append({
    'session': 0, 'known_acc': 0.9367, 'unknown_acc': 0.0,
    'inc_acc': 0.0, 'all_acc': 0.9367,
    'auroc_margin': float('nan'), 'oscr': float('nan'), 'fpr95': float('nan'),
})
for s in range(1, 5):
    d = np.load(os.path.join(D, f'session_{s}_raw.npz'))
    margins = d['margins']; is_unk = d['is_unknown'].astype(int)
    correct = d['correct'].astype(int); pos = d['pos']
    # known/unknown acc 粗估 (fast_eval 单次, 50 样本)
    # 将 margin<=中位数判为 unknown
    thr = float(np.median(margins))
    pred_unk = (margins <= thr).astype(int)
    unk_acc = float(((pred_unk == 1) & (is_unk == 1)).sum() / max(1, (is_unk == 1).sum()))
    kn_acc = float(((pred_unk == 0) & (is_unk == 0) & (correct == 1)).sum() / max(1, (is_unk == 0).sum()))
    try:
        auroc_m = float(roc_auc_score(is_unk, -margins))
    except Exception:
        auroc_m = float('nan')
    try:
        fpr, tpr, _ = roc_curve(is_unk, -margins)
        idx = np.argmin(np.abs(tpr - 0.95))
        fpr95 = float(fpr[idx])
    except Exception:
        fpr95 = float('nan')
    oscr = compute_oscr(-margins, correct, is_unk)
    session_rows.append({
        'session': s, 'known_acc': round(kn_acc, 4),
        'unknown_acc': round(unk_acc, 4),
        'inc_acc': float('nan'), 'all_acc': float('nan'),
        'auroc_margin': round(auroc_m, 4), 'oscr': round(oscr, 4),
        'fpr95': round(fpr95, 4),
    })

# 写 CSV
keys = []
seen = set()
for r in session_rows:
    for k in r.keys():
        if k not in seen:
            seen.add(k); keys.append(k)
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
    w.writeheader()
    for r in session_rows:
        w.writerow(r)
print(f'[OK] wrote {OUT_CSV}')


# ----------------------- 图 A: t-SNE -----------------------
all_pts = np.concatenate([base_after, new_after, fake], axis=0)
labels = (['base'] * NUM_BASE) + (['new'] * new_after.shape[0]) + (['fake'] * NUM_BASE)
tsne = TSNE(n_components=2, perplexity=15, random_state=0, init='pca', learning_rate=200.0)
emb = tsne.fit_transform(all_pts)
plt.figure(figsize=(8, 7))
for lab, col, mk in [('base', 'tab:blue', 'o'), ('new', 'tab:red', '^'), ('fake', 'tab:green', 'x')]:
    idx = [i for i, l in enumerate(labels) if l == lab]
    plt.scatter(emb[idx, 0], emb[idx, 1], c=col, marker=mk, s=28, alpha=0.75, label=lab)
plt.title('Phase A: t-SNE of base/new/fake prototypes (fc_after+fake)')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(D, 'plot_A_proto_tsne.png'), dpi=120)
plt.close()
print('[OK] plot_A_proto_tsne.png')


# ----------------------- 图 B: margin scatter per session -----------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for i, s in enumerate(range(1, 5)):
    d = np.load(os.path.join(D, f'session_{s}_raw.npz'))
    m = d['margins']; cm = d['cls_margins']; iu = d['is_unknown']
    ax = axes[i]
    ax.scatter(m[iu == 0], cm[iu == 0], c='tab:blue', s=32, label='known', alpha=0.75)
    ax.scatter(m[iu == 1], cm[iu == 1], c='tab:red', s=32, label='unknown', alpha=0.75, marker='x')
    ax.axvline(np.median(m), color='gray', ls='--', lw=1, label='median(margin)')
    ax.set_title(f'Session {s}')
    ax.set_xlabel('margin = pos - neg')
    if i == 0:
        ax.set_ylabel('cls_margin')
    ax.legend(fontsize=8)
plt.suptitle('Phase B: per-session margin × cls_margin scatter')
plt.tight_layout()
plt.savefig(os.path.join(D, 'plot_B_margin_scatter.png'), dpi=120)
plt.close()
print('[OK] plot_B_margin_scatter.png')


# ----------------------- 图 C: attention heat map -----------------------
plt.figure(figsize=(8, 7))
plt.imshow(attn80, cmap='viridis', aspect='auto')
plt.colorbar(label='attention weight')
plt.title(f'Phase C: 80-way calibrator attention (norm-ent={ent80_norm:.3f}, top1={top1_80:.3f})')
plt.xlabel('key (base proto idx)')
plt.ylabel('query (base proto idx)')
plt.tight_layout()
plt.savefig(os.path.join(D, 'plot_C_attention80.png'), dpi=120)
plt.close()
print('[OK] plot_C_attention80.png')


# ----------------------- 图 D: 原型几何三联图 -----------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# D1: fc_meta 和 fc_after 的两两相似度分布
ax = axes[0]
off_meta = sim_meta_base[~np.eye(NUM_BASE, dtype=bool)]
off_after = sim_after_base[~np.eye(NUM_BASE, dtype=bool)]
ax.hist(off_meta, bins=40, alpha=0.55, label=f'meta (mean={off_meta.mean():.3f})', color='tab:blue')
ax.hist(off_after, bins=40, alpha=0.55, label=f'after replace (mean={off_after.mean():.3f})', color='tab:orange')
ax.set_title('D1: base pairwise cos (meta vs replaced)')
ax.set_xlabel('cos'); ax.legend()
# D2: pos vs neg 分布 + neg pairwise
ax = axes[1]
ax.hist(pos_vs_neg, bins=30, alpha=0.65, color='tab:green', label=f'pos-vs-neg (mean={pos_vs_neg.mean():.3f})')
off_neg = neg_pair[~np.eye(NUM_BASE, dtype=bool)]
ax.hist(off_neg, bins=30, alpha=0.55, color='tab:red', label=f'neg-pair (mean={off_neg.mean():.3f})')
ax.set_title('D2: dual prototype geometry')
ax.set_xlabel('cos'); ax.legend()
# D3: new 到最近 base 的相似度
ax = axes[2]
ax.hist(min_sim_new2base, bins=20, color='tab:purple', alpha=0.8)
ax.axvline(min_sim_new2base.mean(), color='k', ls='--', label=f'mean={min_sim_new2base.mean():.3f}')
ax.set_title('D3: new-to-nearest-base cos (after compact)')
ax.set_xlabel('cos'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(D, 'plot_D_proto_geometry.png'), dpi=120)
plt.close()
print('[OK] plot_D_proto_geometry.png')


# ----------------------- report.md -----------------------
report = []
report.append('# Diagnostic Report: v3 (layer4 ft, scale=0.02)\n')
report.append('生成源: `/data/lqq/baseline/save_result/diagnose_v3/`  \n')
report.append(f'基座 checkpoint: `save/epoch_15.pth`\n\n')

report.append('## 1. 原型几何 (Phase A)\n')
for k, v in geom.items():
    report.append(f'- `{k}` = **{v:.4f}**\n')

report.append('\n### 诊断结论 1\n')
c1_bad = geom['fc_after_base_pair_cos_mean'] > 0.3
c2_bad = geom['meta_vs_after_same_class_cos_mean'] < 0.8
c3_bad = geom['pos_vs_neg_cos_mean'] > 0.3
c4_bad = geom['new_to_nearest_base_cos_mean'] > 0.6
report.append(f'- **Base 原型 replace 后坍缩**: {"YES ❌" if c1_bad else "no"} '
              f'(meta 时 {geom["fc_meta_base_pair_cos_mean"]:.3f} → replace 后 '
              f'{geom["fc_after_base_pair_cos_mean"]:.3f})\n')
report.append(f'- **同类 meta vs replace 原型偏移大**: {"YES ❌" if c2_bad else "no"} '
              f'(diag cos = {geom["meta_vs_after_same_class_cos_mean"]:.3f}, 应 > 0.8)\n')
report.append(f'- **对偶原型坍缩(pos≈neg)**: {"YES ❌" if c3_bad else "no"} '
              f'(pos-vs-neg cos = {geom["pos_vs_neg_cos_mean"]:.3f}, 应 < 0.3)\n')
report.append(f'- **new 原型被拉进 base 空间**: {"YES ❌" if c4_bad else "no"} '
              f'(new-to-nearest-base cos = {geom["new_to_nearest_base_cos_mean"]:.3f}, 应 < 0.6)\n')

report.append('\n## 2. 规模失配 (Phase C)\n')
report.append('| 输入规模 | 归一化熵 | top1 prob | 评价 |\n|---|---|---|---|\n')
report.append(f'| 5-way (训练分布) | 0.086 | 0.975 | ✅ attention 集中 |\n')
report.append(f'| 80-way (session 0) | 0.439 | 0.645 | ⚠️ 扁平化 5 倍 |\n')
report.append(f'| 100-way (session 4) | 0.534 | 0.518 | ❌ 近似均匀分布 |\n')
report.append('\n**结论**: calibrator 在 80+ 类下 attention 大幅失效, 用户质疑成立: '
              'AttnClassifier 只见过 5-way 训练, 推理时 80~100-way 直接超出分布.\n')

report.append('\n## 3. 特征空间一致性 (Phase B)\n')
report.append('encode vs hgnn_encode+spatial-mean 配对余弦 = 1.0000. **不是问题**.\n')

report.append('\n## 4. Session-level OSR 指标 (CSV)\n')
report.append('| session | AUROC(margin) | OSCR | FPR@TPR95 |\n|---|---|---|---|\n')
for r in session_rows[1:]:
    report.append(f'| {r["session"]} | {r["auroc_margin"]} | {r["oscr"]} | {r["fpr95"]} |\n')
report.append('\n**注意**: fast_eval 单次采样 50 样本, 波动大, 仅供定性参考.\n')

report.append('\n## 5. 核心结论 (回应用户质疑)\n')
report.append(
    '1. **对偶原型机制有效但不够**: neg_pair_cos≈0.22 说明 5 个 neg 原型之间有差异, '
    '但 pos-vs-neg cos≈0.51 说明**对偶对本身没拉开**, 无法为 OSR 提供强可分性.\n'
)
report.append(
    '2. **最大结构缺陷**: `replace_base_fc` 用训练集均值直接覆盖掉 meta_train 学到的 base 原型, '
    f'导致 base 两两相似度从 -0.005 (近似正交) 暴涨到 0.66. 这让 cosine 分类器在 80 类间几乎失去区分度, '
    '是 S3 known_acc=0.578 / inc_acc=0.549 暴跌的直接原因.\n'
)
report.append(
    '3. **规模失配确认**: SupportCalibrator 在 80~100 way 下 attention 归一化熵 0.44~0.53, '
    '已经**近似均匀**, 即 calibrator 等价于把所有 base 原型平均再加到 query 上, **结构性失灵**.\n'
)
report.append(
    '4. **用户核心直觉正确**: base 类是大样本, 不应让只见过 5-way 的 AttnClassifier 同时处理. '
    '正确做法应该是: base 原型用大样本离线校准并绕过 calibrator, AttnClassifier 只对 new 类 5-shot 做适配.\n'
)
report.append(
    '5. **不确定性课程学习影响**: 目前还没有"关闭课程学习"的 A/B 数据, '
    '但 meta-time base 原型 pair-cos=-0.005 显示训练完后 base 空间已很分散, '
    '更可能的退化源是 replace_base_fc, 不是课程学习本身.\n'
)

report.append('\n## 6. 下一步建议 (不在本次诊断中实施)\n')
report.append('- **修复 A (根治)**: 替换 `replace_base_fc` 为更严谨的"原型校准":  \n'
              '  a) 从训练集抽 shot 后先过 `SupportCalibrator`, 再均值, 使其与训练分布一致;  \n'
              '  b) 或者干脆**保留 meta 学到的 fc.weight[:80]**, 只用 compact 更新 new 类\n')
report.append('- **修复 B (架构)**: calibrator 输入解耦: base 原型旁路 (取 fc.weight[:80] 不做 self-attn), '
              '只让 new 类 5 个 proto 参与 AttnClassifier\n')
report.append('- **修复 C (损失)**: OpenSetGenerater.forward_dual_loss 加显式 '
              'pos-vs-neg orthogonality (或 margin loss), 逼 pos-vs-neg cos < 0.2\n')
report.append('- **评测**: 已经把 AUROC/OSCR 接入 sessions.csv, 后续评测一并报告\n')

with open(OUT_REPORT, 'w') as f:
    f.writelines(report)
print(f'[OK] wrote {OUT_REPORT}')
print('DONE')
