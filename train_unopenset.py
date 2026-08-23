import os
    # 环境变量
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import random
import yaml
import torch
from sklearn.preprocessing import RobustScaler
import torch.nn as nn  
from utils.util import cluster_acc,calc
from utils.utils import *
from network import MYNET,get_optimizer,replace_base_fc
from data.dataloader import *
from data.sampler import CategoriesSampler
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
from openmax import *
from models.metatrainer_oo import meta_train
# from models.metaowtrainer import meta_train
from threshold_free import run_test_fsl, reset_session_stats, _adaptive_margin_threshold
from models.AttnClassifier import Classifier
from utils.streamCluster import FStream
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 用于3D可视化
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances  # 
from sklearn.cluster import DBSCAN  # DBSCAN
from matplotlib import rcParams
from sklearn.metrics.pairwise import cosine_similarity 
from scipy.optimize import linear_sum_assignment
from enhance_module import LocalFeatureCluster
from models.robust_proto_adapter import RobustPrototypeAdapter
import math
import joblib


class PrototypeStatisticsMemory:
    """Exemplar-free per-class moments used for uncertainty-aware replay."""
    def __init__(self, base_protos, base_var=0.02):
        p = F.normalize(base_protos.detach(), dim=-1)
        self.mean = {i: p[i].clone() for i in range(len(p))}
        if torch.is_tensor(base_var):
            variance = base_var.detach().to(p.device)
            if variance.ndim == 1:
                variance = variance.unsqueeze(0).expand_as(p)
        else:
            variance = torch.full_like(p, float(base_var))
        self.var = {i: variance[i].clamp(1e-6, 0.05).clone() for i in range(len(p))}
        self.confidence = {i: 1.0 for i in range(len(p))}
        self.enabled = True

    def update_novel(self, cls, features, shrink=0.7):
        z = F.normalize(features.detach(), dim=-1)
        mu = z.mean(0)
        empirical = z.var(0, unbiased=False).clamp_min(1e-5)
        old_ids = sorted(self.mean)
        old_mu = torch.stack([self.mean[i].to(z.device) for i in old_ids])
        old_var = torch.stack([self.var[i].to(z.device) for i in old_ids])
        similarity = old_mu @ F.normalize(mu, dim=0)
        weight = F.softmax(similarity / 0.1, dim=0)
        transferred = (weight[:, None] * old_var).sum(0)
        n = float(len(z))
        reliability = n / (n + 10.0)
        shrink_weight = float(shrink) * (1.0 - reliability)
        cov = (1.0 - shrink_weight) * empirical + shrink_weight * transferred
        self.mean[int(cls)] = F.normalize(mu, dim=0).clone()
        self.var[int(cls)] = cov.clamp(1e-5, 0.2).clone()
        self.confidence[int(cls)] = reliability


_PROTO_STATS_MEMORY = None
_NOVEL_SUPPORT_BANK = {}
_LABELED_SUPPORT_PROTOS = {}
_PRECOMPUTED_DISCOVERY_ASSIGNMENTS = None
_ROBUST_PROTO_ADAPTER = None


def ridge_refine_novel_prototypes(args, model, novel_ids):
    """Discriminatively residualize new prototypes while freezing every old row."""
    if not novel_ids or not bool(getattr(args, 'ridge_novel_refine', False)):
        return
    seen = args.num_labeled_classes + args.way
    device = model.fc.weight.device
    with torch.no_grad():
        anchor = model.fc.weight[:seen].detach().clone()
        x = F.normalize(anchor, dim=-1)
        gram = x @ x.t()
        ridge = float(getattr(args, 'ridge_novel_lambda', 0.1))
        eye = torch.eye(seen, device=device, dtype=x.dtype)
        # W = (XX^T + lambda I)^-1 X is the primal classifier expressed
        # through the much smaller class-anchor Gram matrix.
        discriminative = torch.linalg.solve(gram + ridge * eye, x)
        discriminative = F.normalize(discriminative, dim=-1)
        blend = min(max(float(getattr(args, 'ridge_novel_blend', 0.2)), 0.0), 1.0)
        for cls in novel_ids:
            updated = F.normalize((1.0 - blend) * F.normalize(anchor[cls], dim=0)
                                  + blend * discriminative[cls], dim=0)
            # Preserve the legacy head's row norm even though evaluation uses cosine.
            model.fc.weight.data[cls] = updated * anchor[cls].norm().clamp_min(1e-8)
    print(f'  [RIDGE-NOVEL] classes={novel_ids} lambda={ridge:.4g} blend={blend:.3f} old_frozen=True')


def hard_statistical_replay(args, model, seen_count):
    """Optimize prototypes on moment-replayed hard classes while anchoring history."""
    global _PROTO_STATS_MEMORY
    if _PROTO_STATS_MEMORY is None or not _PROTO_STATS_MEMORY.enabled or seen_count <= 1:
        return
    device = model.fc.weight.device
    ids = list(range(seen_count))
    for i in ids:
        if i not in _PROTO_STATS_MEMORY.mean:
            fallback = F.normalize(model.fc.weight[i].detach(), dim=0)
            _PROTO_STATS_MEMORY.mean[i] = fallback.clone()
            known_var = torch.stack(list(_PROTO_STATS_MEMORY.var.values())).mean(0).to(fallback.device)
            _PROTO_STATS_MEMORY.var[i] = known_var.clamp(1e-6, 0.05).clone()
            _PROTO_STATS_MEMORY.confidence[i] = 0.0
    means = torch.stack([_PROTO_STATS_MEMORY.mean[i].to(device) for i in ids])
    variances = torch.stack([_PROTO_STATS_MEMORY.var[i].to(device) for i in ids])
    confidence = torch.tensor([_PROTO_STATS_MEMORY.confidence[i] for i in ids], device=device)
    per_class = int(getattr(args, 'stat_replay_samples', 16))
    noise = torch.randn(seen_count, per_class, means.size(1), device=device)
    replay = F.normalize(means[:, None] + noise * variances.sqrt()[:, None], dim=-1)
    target = torch.arange(seen_count, device=device)[:, None].expand(-1, per_class).reshape(-1)
    replay = replay.reshape(-1, means.size(1))
    anchor = model.fc.weight[:seen_count].detach().clone()
    proto = anchor.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([proto], lr=float(getattr(args, 'stat_replay_lr', 0.03)), momentum=0.9)
    # Classes with a close competitor and uncertain moments receive more replay weight.
    sim = F.normalize(anchor, dim=-1) @ F.normalize(anchor, dim=-1).t()
    sim.fill_diagonal_(-1.0)
    hardness = F.softmax(sim.max(1).values / max(float(getattr(args, 'stat_hard_temperature', 0.1)), 1e-6), dim=0)
    class_weight = (hardness * seen_count + (1.0 - confidence)).detach()
    for _ in range(int(getattr(args, 'stat_replay_steps', 30))):
        logits = F.normalize(replay, dim=-1) @ F.normalize(proto, dim=-1).t() / 0.07
        ce = F.cross_entropy(logits, target, reduction='none')
        loss_cls = (ce * class_weight[target]).mean()
        anchor_per_class = ((proto - anchor) ** 2).mean(1)
        anchor_scale = 1.0 + (1.0 - confidence)
        loss_anchor = (anchor_per_class * anchor_scale).mean()
        loss = loss_cls + float(getattr(args, 'stat_anchor_weight', 2.0)) * loss_anchor
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        novel_strength = min(max(float(getattr(args, 'stat_update_strength', 0.1)), 0.0), 1.0)
        old_strength = getattr(args, 'stat_old_update_strength', None)
        old_strength = novel_strength if old_strength is None else min(max(float(old_strength), 0.0), 1.0)
        strength = torch.full((seen_count, 1), novel_strength, device=device, dtype=anchor.dtype)
        strength[:args.num_base] = old_strength
        blended = (1.0 - strength) * anchor + strength * proto.detach()
        model.fc.weight.data[:seen_count] = blended
        for i in ids:
            _PROTO_STATS_MEMORY.mean[i] = F.normalize(blended[i], dim=0).clone()

# [新增] 这是一个专门针对不确定性设计的 Center Loss
class UncertaintyCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(UncertaintyCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, class_weights=None):
        batch_size = x.size(0)
        # 计算距离矩阵 (x-c)^2 = x^2 + c^2 - 2xc
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        
        
        # 这里的 input 是 distmat, mat1 是 x, mat2 是 centers.t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        
        if class_weights is not None:
            sample_weights = class_weights[labels[:, 0]] 
            loss = (dist.sum(1) * sample_weights).sum() / batch_size
        else:
            loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
def set_mcd_mode(model):
    """
    开启 MC Dropout 模式：
    保持 BatchNorm 为 eval 模式（稳定统计量），但强制开启 Dropout（引入随机性）。
    """
    model.eval() # 全局设为 eval
    
    # 遍历所有子模块，单独激活 Dropout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
def calculate_uncertainty_unlabeled(model, enhancer, sample, n_aug=5, n_forward=5):
    """
    计算无标签样本的不确定度 (基于特征掩码 + MC Dropout)
    """
    # 1. 开启 MC Dropout 模式 (Dropout 生效)
    set_mcd_mode(model)
    
    features_list = []
    device = next(model.parameters()).device
    
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)
    sample = sample.to(device)

    with torch.no_grad():
        # 外层循环：不同的 Mask (通过 augment=True 触发)
        for _ in range(n_aug):
            # 内层循环：不同的 Dropout (通过 MC Dropout 触发)
            for _ in range(n_forward):
                
                # 【关键】调用时开启 augment=True
                # 这会触发 Log Mel 谱图上的随机时间/频率遮挡
                feat = model.hgnn_encode(sample, augment=True) 
                
                # 通过增强模块
                feat, _ = enhancer(feat) 
                
                if feat.dim() > 2:
                    feat = feat.mean(dim=[2,3]) if feat.dim()==4 else feat.mean(dim=1)
                
                features_list.append(feat.squeeze())
    
    P = torch.stack(features_list)
    
    # 计算核范数
    uncertainty = torch.norm(P, p='nuc').item()
    
    return uncertainty

def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    import os
    
    # 基础种子设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 强制确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_randomness():
    """验证随机种子是否生效"""
    print("\n=== Randomness Check ===")
    print(f"Python random: {random.randint(0, 100)}")
    print(f"Numpy random: {np.random.randint(0, 100)}")
    print(f"PyTorch random: {torch.rand(1).item()}")
    print("="*30)
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(args.seed)  # 为初始化过程设种子
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns


def set_up_datasets(args):
    if args.dataset == 'FMC':
        import data.FMC as Dataset
    elif args.dataset == 'nsynth-100':
        import data.nsynth as Dataset
    elif args.dataset == 'nsynth-200':
        import data.nsynth as Dataset
    elif args.dataset == 'nsynth-300':
        import data.nsynth as Dataset
    elif args.dataset == 'nsynth-400':
        import data.nsynth as Dataset
    elif args.dataset == 'librispeech':
        import data.librispeech as Dataset
    elif args.dataset in ['f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n']:
        import data.s2s as Dataset
    args.Dataset=Dataset

def args_parser():
    parser = argparse.ArgumentParser(description='cluster', add_help=False)
    parser.add_argument('-config', type=str, default="/data/lqq/baseline/configs/default.yml") 
    parser.add_argument('-dist_path', type=str, default="/data/lqq/baseline/save/dist.mat") 
    parser.add_argument('-dataset', type=str, default='librispeech',
                        choices=['FMC', 'nsynth-100', 'nsynth-200', 'nsynth-300', 'nsynth-400', 'librispeech',
                        'f2n', 'f2l', 'n2f', 'n2l', 'l2f', 'l2n'])
    # parser.add_argument('--dataroot', type=str,default="/data/datasets/The_NSynth_Dataset/")
    # parser.add_argument('--dataroot', type=str,default="/data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data")
    
    parser.add_argument('--dataroot', type=str,default="/data/datasets/librispeech_fscil/")
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--save_result',type = str,default='/data/lqq/baseline/save_result/')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional isolated checkpoint directory override for retraining.')
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    parser.add_argument('--num_labeled_classes', default=80, type=int)
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ('yes', 'y', 'true', 't', '1'):
            return True
        if s in ('no', 'n', 'false', 'f', '0', ''):
            return False
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')
    parser.add_argument('--checkpoint', type=_str2bool, default=True)
    parser.add_argument('--load_base', type=_str2bool, default=True, help='Skip base training and load pretrained base model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cosine', type=bool,default=True, help='using cosine annealing')
    parser.add_argument('--pretrained_model_path', type=str, default="/data/lqq/baseline/save/base_train_for_meta.pth")
    parser.add_argument('--train_weight_base', type=int, default=1, help='enable training base class weights')
    parser.add_argument('--base_seman_calib',type=int, default=1, help='base semantics calibration')
    parser.add_argument('--neg_gen_type', type=str, default='att', choices=['semang', 'attg', 'att', 'mlp'])
    parser.add_argument('--agg', type=str, default='avg', choices=['avg', 'mlp'])
    parser.add_argument('--gamma', type=float, default=1.0, help='loss cofficient for mse loss')
    parser.add_argument('--funit', type=float, default=1.0)
    parser.add_argument('--outer_lr', type=float, default=0.001)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--inner_steps', default=10, type=int) 
    parser.add_argument('--outer_steps', default=5, type=int)
    parser.add_argument('--debug', default=True, type=bool)
    parser.add_argument('--use_cffm_eval', type=_str2bool, default=False,
                        help='Use the checkpointed model CFFM during OSR evaluation. '
                             'Disabled by default because legacy checkpoints were not '
                             'trained/evaluated consistently with this path.')
    # 在args_parser()中添加以下参数
    parser.add_argument('--pit_weight', type=float, default=0.5, help='weight for pseudo-incremental loss')
    parser.add_argument('--pit_num_new_classes', type=int, default=5, help='number of pseudo new classes')
    parser.add_argument('--pit_base_momentum', type=float, default=0.7, help='momentum for base class weight update')
    parser.add_argument('--pit_mixup_alpha', type=float, default=0.5, help='alpha for mixup augmentation')
    # ===== tuning hyperparams (Task 1.2 / 1.3 / 1.4) =====
    parser.add_argument('--curriculum_ratio', type=float, default=0.3,
                        help='fraction of total epochs devoted to the fast curriculum stage')
    parser.add_argument('--center_loss_weight', type=float, default=0.005,
                        help='weight for UncertaintyCenterLoss in base training')
    parser.add_argument('--hard_weight_scale', type=float, default=1.0,
                        help='max multiplier for hard-class weighted sampling (1+ratio*scale)')
    parser.add_argument('--base_standard_only', type=_str2bool, default=False,
                        help='train all base classes uniformly for every epoch; used as a '
                             'clean control when uncertainty curriculum hurts a dataset')
    parser.add_argument('--base_val_patience', type=int, default=6,
                        help='Early-stopping patience for base training using the disjoint validation CSV.')
    parser.add_argument('--base_val_min_delta', type=float, default=1e-4,
                        help='Minimum validation-accuracy improvement required to reset patience.')
    parser.add_argument('--base_init_checkpoint', type=str, default='',
                        help='Optional local checkpoint used to initialize base metric fine-tuning.')
    parser.add_argument('--base_finetune_lr', type=float, default=0.0,
                        help='Override base optimizer LR when fine-tuning an existing model.')
    parser.add_argument('--base_supcon_weight', type=float, default=0.0,
                        help='Weight of supervised contrastive loss on current model.encode features.')
    parser.add_argument('--base_supcon_temperature', type=float, default=0.1)
    parser.add_argument('--base_supcon_balanced', type=_str2bool, default=False,
                        help='Use class-balanced 16-class x 8-sample batches for SupCon fine-tuning.')
    parser.add_argument('--hinge_margin', type=float, default=2.0,
                        help='margin for open-set hinge loss between pos/neg prototypes')
    parser.add_argument('--compact_lr', type=float, default=0.05,
                        help='learning rate for prototype compaction optimizer')
    parser.add_argument('--compact_steps', type=int, default=30,
                        help='steps for prototype compaction optimizer')
    parser.add_argument('--prototype_trim_farthest', type=int, default=0,
                        help='Robust pseudo-cluster prototype: iteratively remove this many farthest '
                             'model.encode samples before taking the mean.')
    parser.add_argument('--robust_proto_adapter_path', type=str, default='',
                        help='Local base-only robust prototype adapter checkpoint; empty disables it.')
    parser.add_argument('--prototype_linear_adapter_path', type=str, default='',
                        help='Offline base-only residual map from model.encode means to trained classifier weights.')
    parser.add_argument('--prototype_linear_adapter_strength', type=float, default=1.0,
                        help='Residual interpolation strength for the offline linear prototype adapter.')
    parser.add_argument('--compact_base_margin', type=float, default=0.2,
                        help='minimum cosine distance between new proto and any base proto')
    parser.add_argument('--compact_novel_margin', type=float, default=0.0,
                        help='Z-b: minimum cosine distance between new proto and same-session novel protos (repel within session). 0=disabled.')
    parser.add_argument('--compact_novel_weight', type=float, default=1.0,
                        help='Z-b: weight of novel-novel repel loss in prototype compaction (relative to compact loss).')
    parser.add_argument('--session_thr_blend', type=float, default=0.0,
                        help='blend weight (0-1) for session-aware threshold smoothing; 0=adaptive only')
    parser.add_argument('--kmeans_filter_thr', type=float, default=0.0,
                        help='cos-similarity threshold to base protos for filtering base leakage before KMeans; 0=disabled')
    parser.add_argument('--kmeans_filter_quantile', type=float, default=0.0,
                        help='Adaptive base-leakage filter: retain the lowest-similarity fraction of candidates. '
                             'For example 0.7 retains 70%%; 0 disables it. Applied before the optional fixed threshold.')
    parser.add_argument('--balanced_kmeans', type=_str2bool, default=False,
                        help='Use capacity-constrained assignments after KMeans initialization. This encodes the '
                        'balanced N-way K-shot stream prior and prevents a single merged mega-cluster.')
    parser.add_argument('--cluster_algorithm', choices=['kmeans', 'agglomerative'], default='kmeans',
                        help='CANA grouping algorithm after label-free candidate selection.')
    parser.add_argument('--balanced_kmeans_iters', type=int, default=5,
                        help='Number of constrained assignment/centroid refinement iterations.')
    parser.add_argument('--normalize_cluster_features', type=_str2bool, default=False,
                        help='L2-normalize model.encode features before CANA KMeans (spherical clustering).')
    parser.add_argument('--use_joint_cluster_assignments', type=_str2bool, default=False,
                        help='Reuse the five clusters selected by joint 10-way CANA instead of reclustering them.')
    parser.add_argument('--session_restricted_alignment', type=_str2bool, default=True,
                        help='For permutation-invariant evaluation, align discovered clusters only to the '
                             'five current-session class IDs; never overwrite prior-session prototypes.')
    parser.add_argument('--structure_feature_layers', type=str, default='layer4',
                        help='Comma-separated ResNet stages used by the structure discovery branch. '
                             'Supported values: layer2,layer3,layer4. Each pooled stage is normalized '
                             'before concatenation, enabling controlled single- and multi-layer ablations.')
    parser.add_argument('--discovery_encoder', type=str, default='hgnn_lfc',
                        choices=['hgnn_lfc', 'direct'],
                        help='Feature space used to form novel prototypes. direct uses model.encode '
                             'and is classifier-consistent for base-only checkpoints; hgnn_lfc '
                             'preserves the legacy meta-trained discovery path.')
    parser.add_argument('--encode_tta_views', type=int, default=1,
                        help='Deterministic waveform-shift views averaged through the same model.encode; 1 disables TTA.')
    parser.add_argument('--feature_centering', type=_str2bool, default=False,
                        help='Subtract the mean base prototype from query and prototype features '
                             'before cosine classification (classifier-consistent mean removal).')
    parser.add_argument('--incremental_metric', choices=['cosine', 'euclidean', 'dot'], default='cosine',
                        help='Distance used by the expanded classifier at incremental evaluation.')
    parser.add_argument('--incremental_novel_logit_bias', type=float, default=0.0,
                        help='Subtract a validation-calibrated bias from incremental-class logits; 0 disables it.')
    parser.add_argument('--incremental_novel_logit_scale', type=float, default=1.0,
                        help='Scale novel logits before bias calibration; 1 disables group scaling.')
    parser.add_argument('--incremental_base_logit_scale', type=float, default=1.0,
                        help='Scale base logits before novel-vs-base calibration; 1 disables group scaling.')
    parser.add_argument('--incremental_proto_hubness_weight', type=float, default=0.0,
                        help='Training-free per-prototype CSLS-style hubness penalty for cosine incremental evaluation.')
    parser.add_argument('--incremental_proto_hubness_scope', type=str, default='all',
                        choices=['all', 'base', 'novel'],
                        help='Prototype group receiving the hubness penalty.')
    parser.add_argument('--incremental_proto_hubness_k', type=int, default=8,
                        help='Number of neighbouring prototypes used by the hubness penalty.')
    parser.add_argument('--incremental_group_margin_gate', type=_str2bool, default=False,
                        help='Route between base and novel groups using their label-free top1-top2 cosine margins.')
    parser.add_argument('--incremental_group_margin_bias', type=float, default=0.0,
                        help='Bias added to the novel-group confidence margin.')
    parser.add_argument('--incremental_osr_group_gate', type=_str2bool, default=False,
                        help='Route base/novel predictions with the trained positive-negative OSR margin.')
    parser.add_argument('--incremental_group_router_path', type=str, default='',
                        help='Offline validation-trained logistic router between base and novel groups.')
    parser.add_argument('--incremental_group_router_offset', type=float, default=0.0,
                        help='Validation-calibrated additive offset for the group-router logit.')
    parser.add_argument('--incremental_group_router_soft_scale', type=float, default=0.0,
                        help='Use soft novel-logit calibration by this scale instead of hard group routing; 0 is hard.')
    parser.add_argument('--incremental_tree_router_path', type=str, default='',
                        help='Offline base-only nonlinear router stored with joblib.')
    parser.add_argument('--incremental_tree_router_soft_scale', type=float, default=0.0,
                        help='Tree-router log-odds scale; 0 uses a hard group decision.')
    parser.add_argument('--incremental_radius_power', type=float, default=0.0,
                        help='Normalize angular prototype distance by class radius to this power; 0 disables.')
    parser.add_argument('--oracle_eval_group_gate', type=_str2bool, default=False,
                        help='Diagnostic only: use evaluation labels solely for perfect base/novel routing.')
    parser.add_argument('--incremental_quantile_group_gate', type=_str2bool, default=False,
                        help='Transductive label-free base/novel routing using the expected seen-novel class proportion.')
    parser.add_argument('--incremental_quantile_support_topk', type=int, default=1,
                        help='Top-k model.encode support similarities averaged for quantile routing.')
    parser.add_argument('--incremental_quantile_score', choices=['support_margin', 'novel_max', 'novel_gap', 'margin_gap'],
                        default='support_margin', help='Label-free novelty score used by transductive quantile routing.')
    parser.add_argument('--incremental_sinkhorn_balance', type=_str2bool, default=False,
                        help='Label-free transductive class-prior calibration over all seen classes. '
                             'It estimates additive logit biases from the all-class query stream only; '
                             'evaluation labels are never read.')
    parser.add_argument('--incremental_sinkhorn_temperature', type=float, default=0.05,
                        help='Soft-assignment temperature for transductive class balancing.')
    parser.add_argument('--incremental_sinkhorn_iterations', type=int, default=100,
                        help='Number of column-marginal calibration iterations.')
    parser.add_argument('--incremental_sinkhorn_scope', choices=['class', 'group'], default='class',
                        help='Balance every class, or only the aggregate base/novel group prior.')
    parser.add_argument('--novel_base_projection_strength', type=float, default=0.0,
                        help='Remove this fraction of the base-prototype subspace component from each novel prototype; 0 disables it.')
    parser.add_argument('--novel_bank_classifier', type=_str2bool, default=False,
                        help='Classify novel classes by the maximum cosine similarity to their '
                             'model.encode support embeddings instead of only their mean prototype.')
    parser.add_argument('--novel_bank_topk', type=int, default=1,
                        help='Average the top-k support similarities per novel class; 1 is nearest support.')
    parser.add_argument('--novel_bank_temperature', type=float, default=0.0,
                        help='Softmax temperature for support-bank aggregation; 0 keeps uniform top-k mean.')
    parser.add_argument('--novel_bank_blend', type=float, default=1.0,
                        help='Blend bank score with mean-prototype score; 1 uses bank only.')
    parser.add_argument('--use_pan_incremental', type=_str2bool, default=False,
                        help='Use the checkpoint-trained APGM/PQAM attention path for incremental '
                             'classification, with model.encode support embeddings and query-conditioned prototypes.')
    parser.add_argument('--reset_fc_each_round', type=_str2bool, default=True,
                        help='Reset model state before every independent evaluation repeat. '
                             'Must remain True for publishable results; False leaks novel '
                             'prototypes across repeats and inflates later rounds.')
    parser.add_argument('--skip_meta_train', type=_str2bool, default=False,
                        help='If True, skip meta_train call and use the loaded checkpoint directly (for fast eval-only iteration).')
    parser.add_argument('--checkpoint_name', type=str, default='epoch_5.pth',
                        help='checkpoint filename under save_dir when --checkpoint is set')
    parser.add_argument('--full_checkpoint_path', type=str, default='',
                        help='Optional local base-only checkpoint containing a params state dict. '
                             'When set, it takes precedence over the legacy cls_params checkpoint. '
                             'The path must exist locally; no download fallback is attempted.')
    parser.add_argument('--finetune_encoder', type=_str2bool, default=False,
                        help='Enable encoder.layer4 low-LR fine-tuning during meta_train (v2)')
    parser.add_argument('--encoder_lr_scale', type=float, default=0.01,
                        help='LR scale multiplier for encoder.layer4 when --finetune_encoder=True')
    parser.add_argument('--finetune_layers', type=str, default='layer4',
                        help='Comma-separated encoder layers to fine-tune, e.g. "layer3,layer4"')
    parser.add_argument('--skip_replace_base_fc', type=_str2bool, default=False,
                        help='If True, keep meta-trained fc.weight[:num_base] instead of overwriting with training-set mean (fix for base prototype collapse).')
    parser.add_argument('--orth_base_proto', type=_str2bool, default=False,
                        help='After replace_base_fc, apply ZCA-whitening on fc.weight[:num_base] to decorrelate base prototypes while keeping them in the feature-mean space (path B).')
    parser.add_argument('--orth_strength', type=float, default=1.0,
                        help='Blend factor between original mean-proto and whitened proto: 0=original, 1=fully whitened.')
    parser.add_argument('--dual_cos_weight', type=float, default=0.0,
                        help='Path C: weight of dual-prototype cosine orthogonality loss (pos ⊥ neg + neg diversity) during meta_train. 0=off.')
    parser.add_argument('--dual_cos_margin', type=float, default=0.2,
                        help='Path C: upper bound for cos(pos,neg). Penalize cos > margin.')
    parser.add_argument('--neg_div_margin', type=float, default=0.1,
                        help='Path C: upper bound for pairwise cos between neg prototypes.')
    parser.add_argument('--inter_cos_weight', type=float, default=0.0,
                        help='Path D: weight of inter-class prototype cosine penalty (keep known protos apart). 0=off.')
    parser.add_argument('--inter_cos_margin', type=float, default=0.3,
                        help='Path D: upper bound for pairwise cos between known-class supp protos.')
    parser.add_argument('--osr_margin_weight', type=float, default=0.0,
                        help='Path E: weight of explicit per-sample OSR margin loss (pos-vs-neg decision per sample). 0=off.')
    parser.add_argument('--osr_margin_val', type=float, default=0.5,
                        help='Path E: margin value for OSR decision: known expects pos-neg > val, unknown expects neg-pos > val.')
    parser.add_argument('--base_anchor_weight', type=float, default=0.0,
                        help='Path Y: weight of L2 anchor loss pulling fine-tuned encoder layers back to init (prevent base feature drift). 0=off.')
    parser.add_argument('--proto_ema_alpha', type=float, default=0.0,
                        help='EMA weight for blending existing fc.weight[new_class] with newly clustered proto; '
                             '0.0 = overwrite (baseline), 0.5 = equal blend, 0.7 = prefer history')
    parser.add_argument('--teen_calibration', type=_str2bool, default=False,
                        help='Training-free semantic calibration of few-shot novel prototypes using nearest old prototypes (TEEN-style).')
    parser.add_argument('--teen_alpha', type=float, default=0.9,
                        help='Weight of the raw novel prototype during TEEN-style calibration.')
    parser.add_argument('--teen_topk', type=int, default=8,
                        help='Number of nearest old prototypes used for calibration; 0 uses all base prototypes as in official TEEN.')
    parser.add_argument('--teen_temperature', type=float, default=0.1,
                        help='Softmax temperature for similarity-weighted old-prototype transfer.')
    parser.add_argument('--teen_preserve_norm', type=_str2bool, default=True,
                        help='Preserve raw prototype norm for this repository classifier; False matches official TEEN normalization.')
    parser.add_argument('--joint_proto_refine', type=_str2bool, default=False,
                        help='If True, run a small joint prototype refinement after clustering so seen base and novel prototypes can move together.')
    parser.add_argument('--joint_proto_refine_lr', type=float, default=0.01,
                        help='learning rate for joint prototype refinement')
    parser.add_argument('--joint_proto_refine_steps', type=int, default=10,
                        help='optimization steps for joint prototype refinement')
    parser.add_argument('--joint_proto_refine_anchor_weight', type=float, default=0.1,
                        help='anchor weight that keeps base prototypes close to their pre-refine positions')
    parser.add_argument('--joint_proto_refine_sep_weight', type=float, default=0.1,
                        help='pairwise separation weight during joint prototype refinement')
    parser.add_argument('--joint_proto_refine_sep_margin', type=float, default=0.15,
                        help='upper cosine-similarity margin for joint prototype refinement separation')
    parser.add_argument('--old_proto_adapt', type=_str2bool, default=False,
                        help='After novel prototype expansion, adapt old prototypes with current known samples and repel them from new prototypes.')
    parser.add_argument('--old_proto_adapt_lr', type=float, default=0.01,
                        help='learning rate for old prototype adaptation')
    parser.add_argument('--old_proto_adapt_steps', type=int, default=10,
                        help='optimization steps for old prototype adaptation')
    parser.add_argument('--old_proto_adapt_anchor_weight', type=float, default=1.0,
                        help='anchor weight that keeps old prototypes close to their pre-adapt positions')
    parser.add_argument('--old_proto_adapt_compact_weight', type=float, default=1.0,
                        help='compactness weight that pulls old prototypes with available known samples toward their feature means')
    parser.add_argument('--old_proto_adapt_sep_weight', type=float, default=0.1,
                        help='separation weight that pushes old prototypes away from newly expanded prototypes')
    parser.add_argument('--old_proto_adapt_sep_margin', type=float, default=0.35,
                        help='upper cosine-similarity margin between old and new prototypes during adaptation')
    parser.add_argument('--old_proto_adapt_guard', type=_str2bool, default=True,
                        help='If True, restore old prototypes when adaptation lowers current known accuracy.')
    parser.add_argument('--osr_thr_scale', type=float, default=1.0,
                        help='scale factor for OSR thresholds (thr_open/thr_cls). '
                             '<1.0 => more samples kept as known; >1.0 => more rejected to unknown')
    parser.add_argument('--support_proto_blend', type=float, default=1.0,
                        help='weight on few-shot model.encode prototype; remaining weight uses current-encoder discovered geometry')
    parser.add_argument('--use_uop', type=_str2bool, default=False,
                        help='Enable Unified Open-set Prototype: chunked OpenSetGenerater + mean → single global open prototype '
                             '(fixes train/test scale mismatch on nway).')
    parser.add_argument('--uop_chunk_size', type=int, default=5,
                        help='chunk size (=training nway) when running OpenSetGenerater over known classes.')
    parser.add_argument('--uop_mode', type=str, default='antimean',
                        choices=['chunk', 'antimean', 'anti_nn'],
                        help='How to derive the global open-set prototype. '
                             'chunk: via OpenSetGenerater (degenerates under neg_gen_type=att). '
                             'antimean: -mean of L2-normalized known protos (recommended default). '
                             'anti_nn: antimean with span projection subtracted (further push-away).')
    parser.add_argument('--osr_noise_std', type=float, default=0.1,
                        help='Noise std injected to supp_protos in incre_forward (non-UOP path) '
                             'to match training distribution and prevent att-residual collapse. '
                             '0.0 = original behavior (fake≈supp, margins≈0). '
                             'Default 0.1 matches Classifier.forward training noise level.')
    parser.add_argument('--osr_use_cls_margin', type=_str2bool, default=False,
                        help='If True, OSR uses margin AND cls_margin; '
                             'if False (default), use margin only (pos-neg at argmax).')
    parser.add_argument('--train_noise_std', type=float, default=0.1,
                        help='Noise std on supp_protos before open_generator in training '
                             '(matches osr_noise_std, helps generator generalize to novel protos).')
    parser.add_argument('--oracle_cluster', type=_str2bool, default=False,
                        help='If True, skip KMeans and use ground-truth labels for novel-class prototype assignment '
                        '(upper-bound experiment).')
    parser.add_argument('--structure_discovery_checkpoint', type=str, default='',
                        help='Frozen DFSB checkpoint used only to augment the clustering space.')
    parser.add_argument('--structure_discovery_weight', type=float, default=0.0,
                        help='Weight of normalized DFSB features concatenated for novel clustering.')
    parser.add_argument('--cluster_all_candidates', type=_str2bool, default=False,
                        help='Retain both rejected and low-confidence accepted stream samples in the discovery buffer.')
    parser.add_argument('--discovery_ranked_topk', type=int, default=0,
                        help='Select this many lowest-knownness stream samples before CANA. '
                             'For balanced N-way K-shot streams use N*K; labels are never used.')
    parser.add_argument('--discovery_rank_start_session', type=int, default=1,
                        help='First incremental session where cluster_all_candidates/ranked selection is active.')
    parser.add_argument('--discovery_rank_score', choices=['osr_margin', 'encode_maxlogit', 'encode_energy',
                                                           'encode_mindist', 'encode_joint_cluster',
                                                           'encode_joint_stats', 'encode_joint_cosine'],
                        default='osr_margin',
                        help='Label-free score for candidate ranking. encode_* uses the current model.encode and '
                             'its trained base classifier, keeping discovery and prototype spaces consistent.')
    parser.add_argument('--base_geometry_path', type=str, default='',
                        help='Local offline validation geometry produced by build_fsc_base_stats.py.')
    parser.add_argument('--joint_cluster_layer', choices=['layer2', 'layer3', 'layer4', 'layer4_lda',
                                                          'layer4_layer3'], default='layer4',
                        help='Stage from the same trained encoder used for joint CANA structure clustering. '
                             'Novel prototypes are always generated by final model.encode features.')
    parser.add_argument('--joint_structure_weight', type=float, default=0.2,
                        help='Layer3 residual weight for layer4_layer3 joint clustering.')
    parser.add_argument('--joint_kmeans_trials', type=int, default=1,
                        help='Deterministic balanced joint-clustering trials selected by label-free silhouette score.')
    parser.add_argument('--joint_kmeans_random_state', choices=['seeded', 'legacy_none'], default='seeded',
                        help='seeded is required for formal runs; legacy_none reproduces historical screening only.')
    parser.add_argument('--discovery_dump_dir', type=str, default='',
                        help='Optional local directory for fixed-seed discovery feature diagnostics.')
    parser.add_argument('--joint_margin_weight', type=float, default=0.0,
                        help='Weight on top1-top2 base-center cosine margin when selecting known/new joint clusters.')
    parser.add_argument('--discovery_reflow_quantile', type=float, default=0.0,
                        help='Fraction of boundary-nearest accepted samples to reflow into discovery; 0 disables it.')
    parser.add_argument('--discovery_reflow_max', type=int, default=10,
                        help='Maximum accepted samples reflowed per session to bound old-class contamination.')
    parser.add_argument('--discovery_reflow_start', type=int, default=1,
                        help='First incremental session eligible for accepted-sample reflow.')
    parser.add_argument('--discovery_reflow_end', type=int, default=999,
                        help='Last incremental session eligible for accepted-sample reflow.')
    parser.add_argument('--discovery_reflow_reject_min', type=float, default=0.0,
                        help='Minimum rejected fraction required for reliability-gated reflow.')
    parser.add_argument('--discovery_reflow_reject_max', type=float, default=1.0,
                        help='Maximum rejected fraction allowed for reliability-gated reflow.')
    parser.add_argument('--mixed_openworld_stream', type=_str2bool, default=False,
                        help='Use a balanced 5-known/5-novel stream episode instead of novel-only support.')
    parser.add_argument('--opt_version', type=str, default='opt_v5',
                        help='sub-directory under save_result for this run')
    parser.add_argument('--run_tag', type=str, default='',
                        help='optional run tag to append to result filename')
    parser.add_argument('--eval_repeats', type=int, default=0,
                        help='Override config test_times when positive.')
    parser.add_argument('--stat_memory', type=_str2bool, default=False,
                        help='Enable uncertainty-aware class statistics memory and hard replay.')
    parser.add_argument('--stat_base_var', type=float, default=-1.0,
                        help='Base normalized diagonal variance; negative estimates it from labeled base features.')
    parser.add_argument('--stat_cov_shrink', type=float, default=0.7,
                        help='Novel covariance shrinkage weight toward similar old classes.')
    parser.add_argument('--stat_replay_samples', type=int, default=16,
                        help='Synthetic samples drawn per seen class for hard replay.')
    parser.add_argument('--stat_replay_steps', type=int, default=30,
                        help='Prototype optimization steps per incremental session.')
    parser.add_argument('--stat_replay_lr', type=float, default=0.03,
                        help='Learning rate for hard statistical replay.')
    parser.add_argument('--stat_anchor_weight', type=float, default=2.0,
                        help='Anchor strength protecting old prototypes during replay.')
    parser.add_argument('--stat_update_strength', type=float, default=0.1,
                        help='Residual write-back strength for replayed prototypes (0-1).')
    parser.add_argument('--stat_old_update_strength', type=float, default=None,
                        help='Separate write-back strength for base/old classes; defaults to stat_update_strength.')
    parser.add_argument('--stat_min_base_acc', type=float, default=0.7,
                        help='Disable synthetic replay when base reliability is below this fixed threshold.')
    parser.add_argument('--stat_min_variance', type=float, default=1e-4,
                        help='Disable replay when cached base moments collapse below this mean variance.')
    parser.add_argument('--stat_hard_temperature', type=float, default=0.1,
                        help='Temperature for confusion-aware replay weighting.')
    parser.add_argument('--ridge_novel_refine', type=_str2bool, default=False,
                        help='Closed-form ridge residualization for session-new prototypes only.')
    parser.add_argument('--ridge_novel_lambda', type=float, default=0.1)
    parser.add_argument('--ridge_novel_blend', type=float, default=0.2)
    # parser.add_argument('--cluster_threshold', type=float, default=0.7, 
    #                   help='Initial threshold for dynamic clustering')
    # parser.add_argument('--threshold_decay', type=float, default=0.95,
    #                   help='Decay rate for cluster threshold')
    # parser.add_argument('--proto_momentum', type=float, default=0.3,
    #                   help='动量系数用于原型更新')
    # parser.add_argument('--debug', action='store_true', 
    #                   help='Enable debug mode with visualizations')
    return parser

def update_fc_avg(args,model,dataloader,x,label,class_list):
    new_fc=[]
    for batch in dataloader:
        x, label,_ = [_.cuda() for _ in batch]
        data=model(x).detach()
    for class_index in class_list:
        print(class_index)
        data_index=(label==class_index).nonzero().squeeze(-1)
        embedding=data[data_index]
        proto=embedding.mean(0)
        new_fc.append(proto)
        if class_index>=args.num_labeled_classes:   #要计算更新这个数
            model.fc.weight.data[class_index]=proto
        else:
            model.fc.weight.data[class_index]=(proto+model.fc.weight.data[class_index]).mean(0)
        #print(proto)
import time  # 需导入时间模块
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
# [添加到 baseline/utils/util.py 或直接放在 train_unopenset.py]
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn.functional as F


def encode_with_deterministic_tta(model, waveform, views=1):
    """Average deterministic time-shift views using only the current model.encode."""
    views = max(int(views), 1)
    embeddings = [model.encode(waveform)]
    if views > 1:
        shift = max(int(waveform.shape[-1] // 100), 1)
        offsets = [shift, -shift, 2 * shift, -2 * shift]
        for offset in offsets[:views - 1]:
            embeddings.append(model.encode(torch.roll(waveform, shifts=offset, dims=-1)))
    return torch.stack(embeddings, dim=0).mean(0)


def debug_cluster(args, model, data, labels, session=None):
    global _PROTO_STATS_MEMORY, _PRECOMPUTED_DISCOVERY_ASSIGNMENTS, _LABELED_SUPPORT_PROTOS
    """
    改进版：聚类 -> 伪标签 -> 原地压缩 (Prototype Compaction)
    Task 1.3 新增：加入 base-proto 余弦距离下限约束，防止新类原型被压向 base。
    Task 1.4 新增：在 KMeans 前过滤掉与 base 原型 cos 相似度过高的 "疑似 base" 样本，
                    避免漏过的 base 污染新类聚类中心。
    oracle_cluster: 跳过 KMeans，直接用真实标签（偷看标签测上限）。
    """
    if data is None or len(data) == 0:
        return 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 提取特征 — 使用 hgnn_encode + 缓存的 LocalFeatureCluster (v10_y 已验证路径)
    with torch.no_grad():
        model.mode = 'incre'
        features_list = []
        structure_features_list = []
        discovery_encoder = str(getattr(args, 'discovery_encoder', 'hgnn_lfc'))
        lfc = _get_lfc(device, k_ratio=0.4) if discovery_encoder == 'hgnn_lfc' else None
        structure_model = _get_structure_discovery_model(args, device)
        for x in data:
            if discovery_encoder == 'direct':
                f = encode_with_deterministic_tta(model, x.to(device),
                                                  int(getattr(args, 'encode_tta_views', 1)))
                features_list.append(f.squeeze())
            else:
                f = model.hgnn_encode(x)           # [1, C, H, W]
                f, _ = lfc(f)                      # LFC (weights cached across sessions)
                features_list.append(f.squeeze())  # [512]
            if structure_model is not None:
                requested_layers = [s.strip() for s in
                                    str(getattr(args, 'structure_feature_layers', 'layer4')).split(',')
                                    if s.strip()]
                pooled_stages = []
                for stage in requested_layers:
                    sf = structure_model.forward_to_stage(x.to(device), stage=stage, augment=False)
                    sf = F.adaptive_avg_pool2d(sf, 1).flatten(1)
                    pooled_stages.append(F.normalize(sf, dim=-1))
                structure_features_list.append(torch.cat(pooled_stages, dim=-1).squeeze())
        features = torch.stack(features_list).to(device)  # [N, 512]
        cluster_features = features
        if structure_features_list:
            structure_features = torch.stack(structure_features_list).to(device)
            weight = float(getattr(args, 'structure_discovery_weight', 0.0))
            cluster_features = torch.cat([
                F.normalize(features, dim=-1),
                weight * F.normalize(structure_features, dim=-1),
            ], dim=-1)
        elif getattr(args, 'normalize_cluster_features', False):
            cluster_features = F.normalize(features, dim=-1)

    # ===== ORACLE PATH: skip KMeans, use ground truth labels directly =====
    if getattr(args, 'oracle_cluster', False):
        global _NOVEL_SUPPORT_BANK
        labels_np = np.array(labels)
        _nl = args.num_labeled_classes
        novel_labels = sorted(set(l for l in labels_np if l >= _nl))
        expected_novel = args.num_unlabeled_classes

        base_end = _nl
        base_protos = model.fc.weight[:base_end, :].detach().to(device)

        print(f'  [ORACLE] session={session} total_unknown={len(labels)} '
              f'novel_found={len(novel_labels)}/{expected_novel} '
              f'novel_ids={novel_labels}')

        for cls_id in novel_labels:
            indices = [i for i, l in enumerate(labels_np) if l == cls_id]
            cls_features = features[indices]
            # Retain current model.encode supports for optional routing/bank
            # scoring. The expanded class prototype itself remains their mean.
            _NOVEL_SUPPORT_BANK[int(cls_id)] = cls_features.detach().clone()
            init_proto = cls_features.mean(dim=0)
            linear_adapter_path = str(getattr(args, 'prototype_linear_adapter_path', ''))
            if linear_adapter_path:
                if not os.path.isfile(linear_adapter_path):
                    raise FileNotFoundError(f'prototype linear adapter is not local: {linear_adapter_path}')
                adapter_state = torch.load(linear_adapter_path, map_location=device, weights_only=True)
                mapped_proto = init_proto @ adapter_state['weight'].to(init_proto)
                strength = float(getattr(args, 'prototype_linear_adapter_strength', 1.0))
                init_proto = init_proto + strength * (mapped_proto - init_proto)

            final_proto = optimize_prototype_compactness(
                init_proto, cls_features,
                lr=getattr(args, 'compact_lr', 0.05),
                steps=getattr(args, 'compact_steps', 30),
                base_protos=base_protos,
                base_margin=getattr(args, 'compact_base_margin', 0.2),
            )
            model.fc.weight.data[cls_id] = final_proto

        return len(novel_labels) / max(expected_novel, 1)
    # ===== END ORACLE =====

    # 收集当前 base (+之前更新的增量类) 原型用于约束和过滤
    base_end = args.num_labeled_classes  # 当前已登记的类数
    base_protos = model.fc.weight[:base_end, :].detach().to(device)

    # --- Task 1.4: 过滤疑似 base 的样本（自适应阈值） ---
    keep_mask = torch.ones(features.shape[0], dtype=torch.bool, device=device)
    kmeans_filter_thr = float(getattr(args, 'kmeans_filter_thr', 0.0))
    kmeans_filter_quantile = float(getattr(args, 'kmeans_filter_quantile', 0.0))
    adaptive_thr = None
    if (kmeans_filter_thr > 0.0 or 0.0 < kmeans_filter_quantile < 1.0) and base_protos.numel() > 0:
        feat_norm = F.normalize(features, dim=-1)
        proto_norm = F.normalize(base_protos, dim=-1)
        cos_to_base = feat_norm @ proto_norm.t()        # [N, base_end]
        max_cos, _ = cos_to_base.max(dim=1)              # [N]
        if 0.0 < kmeans_filter_quantile < 1.0:
            adaptive_thr = torch.quantile(max_cos, kmeans_filter_quantile)
            keep_mask &= max_cos <= adaptive_thr
        if kmeans_filter_thr > 0.0:
            keep_mask &= max_cos < kmeans_filter_thr
        # 保障 kmeans 至少有 n_clusters*2 个样本
        n_clusters = args.num_unlabeled_classes
        if int(keep_mask.sum().item()) < n_clusters * 2:
            keep_mask = torch.ones(features.shape[0], dtype=torch.bool, device=device)
            print('  [CLU-DIAG] candidate filter reverted: fewer than 2 samples per cluster')
        elif adaptive_thr is not None:
            print(f'  [CLU-DIAG] adaptive candidate threshold={float(adaptive_thr):.4f} '
                  f'quantile={kmeans_filter_quantile:.3f} kept={int(keep_mask.sum())}/{features.shape[0]}')

    kept_features = features[keep_mask]
    kept_cluster_features = cluster_features[keep_mask]
    kept_labels_np = np.array(labels)[keep_mask.cpu().numpy()] if len(labels) == features.shape[0] else np.array(labels)

    # 2. 聚类生成伪标签
    n_clusters = args.num_unlabeled_classes
    if kept_features.shape[0] < n_clusters:
        return 0.0

    cluster_np = kept_cluster_features.cpu().numpy()
    use_precomputed = (bool(getattr(args, 'use_joint_cluster_assignments', False)) and
                       _PRECOMPUTED_DISCOVERY_ASSIGNMENTS is not None and
                       len(_PRECOMPUTED_DISCOVERY_ASSIGNMENTS) == len(cluster_np))
    if use_precomputed:
        from types import SimpleNamespace
        pre_labels = np.asarray(_PRECOMPUTED_DISCOVERY_ASSIGNMENTS, dtype=np.int64)
        pre_centers = np.stack([cluster_np[pre_labels == ci].mean(0) for ci in range(n_clusters)])
        kmeans = SimpleNamespace(labels_=pre_labels, cluster_centers_=pre_centers)
        print('  [CLU-DIAG] reusing label-free joint CANA assignments; no second KMeans')
        _PRECOMPUTED_DISCOVERY_ASSIGNMENTS = None
    elif str(getattr(args, 'cluster_algorithm', 'kmeans')) == 'agglomerative':
        try:
            agg = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        except TypeError:
            agg = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average')
        agg_labels = agg.fit_predict(cluster_np)
        from types import SimpleNamespace
        agg_centers = np.stack([cluster_np[agg_labels == ci].mean(0) for ci in range(n_clusters)])
        kmeans = SimpleNamespace(labels_=agg_labels, cluster_centers_=agg_centers)
        print('  [CLU-DIAG] agglomerative average-linkage cosine')
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init=20,
                        random_state=int(getattr(args, 'seed', 0)) + int(session or 0)).fit(cluster_np)
    if getattr(args, 'balanced_kmeans', False) and not use_precomputed:
        # Expand every centroid into a fixed number of assignment slots, then solve
        # the global minimum-cost point-to-slot matching. Capacities differ by at
        # most one, so this remains valid when OSR misses or leaks a few examples.
        n_points = cluster_np.shape[0]
        capacities = np.full(n_clusters, n_points // n_clusters, dtype=np.int64)
        capacities[:n_points % n_clusters] += 1
        centers = kmeans.cluster_centers_.copy()
        balanced_labels = kmeans.labels_.copy()
        for _ in range(max(1, int(getattr(args, 'balanced_kmeans_iters', 5)))):
            slot_classes = np.repeat(np.arange(n_clusters), capacities)
            slot_centers = centers[slot_classes]
            costs = ((cluster_np[:, None, :] - slot_centers[None, :, :]) ** 2).sum(axis=-1)
            row_ind, col_ind = linear_sum_assignment(costs)
            balanced_labels[row_ind] = slot_classes[col_ind]
            new_centers = centers.copy()
            for ci in range(n_clusters):
                mask = balanced_labels == ci
                if np.any(mask):
                    new_centers[ci] = cluster_np[mask].mean(axis=0)
            if np.array_equal(new_centers, centers):
                break
            centers = new_centers
        kmeans.labels_ = balanced_labels
        kmeans.cluster_centers_ = centers
        print(f'  [CLU-DIAG] balanced capacities={capacities.tolist()}')
    pseudo_labels = torch.from_numpy(kmeans.labels_).long().to(device)

    # ===== 聚类诊断: KMeans 匈牙利匹配前的原始纯度 + 每簇真实标签分布 =====
    _nl_clu = args.num_labeled_classes
    _all_in_unknow = len(labels)
    _true_unk_in_input = sum(1 for l in labels if l >= _nl_clu)
    _true_known_in_input = _all_in_unknow - _true_unk_in_input
    print(f'  [CLU-DIAG] Input to KMeans: total={_all_in_unknow} '
          f'true_unknown={_true_unk_in_input} true_known_leaked={_true_known_in_input}')
    if kept_features.shape[0] < len(labels):
        print(f'  [CLU-DIAG] kmeans_filter_thr removed {len(labels) - kept_features.shape[0]} samples')
    # 每簇真实标签分布（匈牙利匹配前）
    _cluster_label_counts = {}
    for ci in range(n_clusters):
        _mask = kmeans.labels_ == ci
        _true_labs = kept_labels_np[_mask]
        _unique, _counts = np.unique(_true_labs, return_counts=True)
        _sorted = sorted(zip(_unique, _counts), key=lambda x: -x[1])
        _top3 = ', '.join(f'cls{l}={c}' for l, c in _sorted[:3])
        _total_c = int(_mask.sum())
        _cluster_label_counts[ci] = _total_c
        print(f'  [CLU-DIAG] cluster_{ci}: size={_total_c} top_labels=[{_top3}]')
    # ===== 聚类诊断结束 =====

    # A detector can reject only leaked old-class samples (observed on FSC-89).
    # The evaluation-only Hungarian helper assumes at least one true novel label;
    # treat this as zero discovery instead of crashing the whole repeated run.
    if not np.any(kept_labels_np >= args.num_labeled_classes):
        print('  [CLU-DIAG] no true novel candidate survived OSR; discovery_acc=0, skip prototype update')
        return 0.0

    # In the actual few-shot setting, map discovered clusters to class IDs using
    # the labeled support episode encoded by the current model.  Never use the
    # evaluation-stream labels for this mapping: ``labels`` above are retained
    # only for diagnostics/metric reporting.
    support_map = {}
    if _LABELED_SUPPORT_PROTOS and str(getattr(args, 'discovery_encoder', 'hgnn_lfc')) == 'direct':
        support_ids = sorted(int(k) for k in _LABELED_SUPPORT_PROTOS
                             if int(k) >= int(args.num_labeled_classes))
        if support_ids:
            support = F.normalize(torch.stack([
                _LABELED_SUPPORT_PROTOS[k].to(device).mean(dim=0)
                if _LABELED_SUPPORT_PROTOS[k].to(device).dim() > 1
                else _LABELED_SUPPORT_PROTOS[k].to(device)
                for k in support_ids]), dim=1)
            centers = F.normalize(torch.stack([
                features[kmeans.labels_ == ci].mean(dim=0) for ci in range(n_clusters)
            ]), dim=1)
            similarity = centers @ support.t()
            rows, cols = linear_sum_assignment((-similarity).detach().cpu().numpy())
            support_map = {int(row): int(support_ids[col]) for row, col in zip(rows, cols)}
            print(f'  [SUPPORT-ALIGN] support_ids={support_ids} '
                  f'mapped={support_map} (evaluation labels excluded)')
    if support_map:
        map_dict = support_map
        # This is an evaluation metric only: labels are not used to construct
        # support_map, but may be used here to report discovery accuracy.
        mapped_labels = np.asarray([map_dict.get(int(c), -1) for c in kmeans.labels_])
        eval_labels = np.asarray(kept_labels_np)
        matched = int(np.sum(mapped_labels == eval_labels))
        acc = matched / max(len(eval_labels), 1)
    elif bool(getattr(args, 'session_restricted_alignment', True)) and session is not None:
        current_start = int(args.num_labeled_classes)
        current_ids = np.arange(current_start, current_start + n_clusters)
        contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
        for cluster_id, class_id in zip(kmeans.labels_, kept_labels_np):
            if int(class_id) in current_ids:
                contingency[int(cluster_id), int(class_id) - current_start] += 1
        rows, cols = linear_sum_assignment(-contingency)
        map_dict = {int(row): int(current_ids[col]) for row, col in zip(rows, cols)}
        matched = int(contingency[rows, cols].sum())
        acc = matched / max(len(kept_labels_np), 1)
        print(f'  [ALIGN] current_ids={current_ids.tolist()} matched={matched}/{len(kept_labels_np)}')
    else:
        acc, map_dict = cluster_acc(args, kept_labels_np, kmeans.labels_)
    nmi = normalized_mutual_info_score(kept_labels_np, kmeans.labels_)
    ari = adjusted_rand_score(kept_labels_np, kmeans.labels_)
    print(f'  [CLU-METRIC] session={session} ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}')

    # 3. 【核心】利用伪标签进行原型炼化
    # Z-b: track session-new protos so later clusters repel from already-updated novel protos in same session
    session_new_protos = []
    session_new_ids = []
    novel_margin_z = float(getattr(args, 'compact_novel_margin', 0.0))
    novel_weight_z = float(getattr(args, 'compact_novel_weight', 1.0))
    for cluster_id in range(n_clusters):
        if cluster_id in map_dict:
            target_class_id = map_dict[cluster_id]
            if target_class_id >= args.num_labeled_classes:

                # 选出该类的样本
                idxs = (pseudo_labels == cluster_id)
                if idxs.sum() == 0: continue
                cluster_feats = kept_features[idxs]

                # Once the cluster has been identified by the labeled few-shot
                # episode, the classifier prototype must come from those current
                # model.encode supports—not from unlabeled test-stream points.
                # The discovered cluster remains useful for diagnostics only.
                support_proto = _LABELED_SUPPORT_PROTOS.get(int(target_class_id))
                if support_proto is not None and str(getattr(args, 'discovery_encoder', 'hgnn_lfc')) == 'direct':
                    support_proto = support_proto.to(device)
                    support_bank = support_proto.unsqueeze(0) if support_proto.dim() == 1 else support_proto
                    # Keep the few-shot model.encode prototype as the anchor, while
                    # allowing a configurable small contribution from the discovered
                    # current-encoder geometry for robustness under acoustic shift.
                    blend = float(getattr(args, 'support_proto_blend', 1.0))
                    if blend < 1.0 and cluster_feats.numel() > 0:
                        discovered = F.normalize(cluster_feats.mean(dim=0, keepdim=True), dim=1)
                        anchor = F.normalize(support_bank.mean(dim=0, keepdim=True), dim=1)
                        support_bank = F.normalize(blend * anchor + (1.0 - blend) * discovered, dim=1)
                    _NOVEL_SUPPORT_BANK[int(target_class_id)] = support_bank.detach().clone()
                    cluster_feats = support_bank
                    print(f'  [PROTO-SUPPORT] class={target_class_id} '
                          f'shots={support_bank.shape[0]} source=model.encode')

                # Do not overwrite the labeled model.encode support bank above.
                # The discovered cluster is retained only for diagnostics and
                # clustering statistics; replacing the bank here would silently
                # turn the real non-oracle path back into an unlabeled-stream
                # prototype and invalidate the support-prototype protocol.
                if _LABELED_SUPPORT_PROTOS.get(int(target_class_id)) is None:
                    _NOVEL_SUPPORT_BANK[int(target_class_id)] = cluster_feats.detach().clone()

                trim_count = min(int(getattr(args, 'prototype_trim_farthest', 0)),
                                 max(int(cluster_feats.shape[0]) - 2, 0))
                for _ in range(trim_count):
                    provisional = F.normalize(cluster_feats.mean(0, keepdim=True), dim=1)
                    similarity = F.normalize(cluster_feats, dim=1) @ provisional.t()
                    remove_idx = int(similarity.squeeze(1).argmin().item())
                    keep_idx = torch.ones(cluster_feats.shape[0], dtype=torch.bool, device=device)
                    keep_idx[remove_idx] = False
                    cluster_feats = cluster_feats[keep_idx]
                if trim_count:
                    print(f'  [PROTO-TRIM] cluster={cluster_id} removed={trim_count} '
                          f'retained={cluster_feats.shape[0]}')

                # 初始化原型：默认均值；可选 base-only 伪簇去噪网络。
                adapter_path = str(getattr(args, 'robust_proto_adapter_path', ''))
                if adapter_path:
                    global _ROBUST_PROTO_ADAPTER
                    if _ROBUST_PROTO_ADAPTER is None:
                        if not os.path.isfile(adapter_path):
                            raise FileNotFoundError(f'robust prototype adapter is not local: {adapter_path}')
                        adapter_state = torch.load(adapter_path, map_location=device, weights_only=True)
                        _ROBUST_PROTO_ADAPTER = RobustPrototypeAdapter(
                            dim=int(adapter_state.get('dim', cluster_feats.shape[1]))).to(device)
                        _ROBUST_PROTO_ADAPTER.load_state_dict(adapter_state['state_dict'], strict=True)
                        _ROBUST_PROTO_ADAPTER.eval()
                    with torch.no_grad():
                        init_proto = _ROBUST_PROTO_ADAPTER(cluster_feats)
                else:
                    init_proto = cluster_feats.mean(dim=0)

                linear_adapter_path = str(getattr(args, 'prototype_linear_adapter_path', ''))
                if linear_adapter_path:
                    if not os.path.isfile(linear_adapter_path):
                        raise FileNotFoundError(f'prototype linear adapter is not local: {linear_adapter_path}')
                    adapter_state = torch.load(linear_adapter_path, map_location=device, weights_only=True)
                    mapped_proto = init_proto @ adapter_state['weight'].to(init_proto)
                    strength = float(getattr(args, 'prototype_linear_adapter_strength', 1.0))
                    init_proto = init_proto + strength * (mapped_proto - init_proto)

                # Z-b: build extra repel set = session_new_protos collected so far
                novel_protos_extra = None
                if len(session_new_protos) > 0 and novel_margin_z > 0.0:
                    novel_protos_extra = torch.stack(session_new_protos).to(device)

                # --- 启动压缩机 (受控版) ---
                final_proto = optimize_prototype_compactness(
                    init_proto, cluster_feats,
                    lr=getattr(args, 'compact_lr', 0.05),
                    steps=getattr(args, 'compact_steps', 30),
                    base_protos=base_protos,
                    base_margin=getattr(args, 'compact_base_margin', 0.2),
                    novel_protos=novel_protos_extra,
                    novel_margin=novel_margin_z,
                    novel_weight=novel_weight_z,
                )
                if getattr(args, 'teen_calibration', False):
                    final_proto = calibrate_novel_prototype(
                        final_proto, base_protos,
                        alpha=float(getattr(args, 'teen_alpha', 0.9)),
                        topk=int(getattr(args, 'teen_topk', 8)),
                        temperature=float(getattr(args, 'teen_temperature', 0.1)),
                        preserve_norm=bool(getattr(args, 'teen_preserve_norm', True)),
                    )
                projection_strength = float(getattr(args, 'novel_base_projection_strength', 0.0))
                if projection_strength > 0.0 and base_protos is not None and base_protos.numel() > 0:
                    with torch.no_grad():
                        original_norm = final_proto.norm().clamp_min(1e-8)
                        basis = torch.linalg.qr(base_protos.t().float(), mode='reduced').Q
                        projected = basis @ (basis.t() @ final_proto.float())
                        final_proto = final_proto.float() - projection_strength * projected
                        final_proto = F.normalize(final_proto, dim=0) * original_norm
                session_new_protos.append(final_proto.detach().clone())
                session_new_ids.append(int(target_class_id))

                # 更新模型（支持 EMA 融合以利用跨 test_time 的原型历史）
                ema_alpha = float(getattr(args, 'proto_ema_alpha', 0.0))
                if ema_alpha > 0.0:
                    with torch.no_grad():
                        prev_proto = model.fc.weight.data[target_class_id].detach()
                        # 若历史原型几乎为零则视为首次初始化，直接覆盖
                        if prev_proto.abs().sum() < 1e-8:
                            model.fc.weight.data[target_class_id] = final_proto
                        else:
                            model.fc.weight.data[target_class_id] = (
                                ema_alpha * prev_proto + (1.0 - ema_alpha) * final_proto
                            )
                else:
                    model.fc.weight.data[target_class_id] = final_proto

                if bool(getattr(args, 'stat_memory', False)) and _PROTO_STATS_MEMORY is not None:
                    _PROTO_STATS_MEMORY.update_novel(
                        target_class_id, cluster_feats,
                        shrink=float(getattr(args, 'stat_cov_shrink', 0.7)))

    ridge_refine_novel_prototypes(args, model, session_new_ids)

    if bool(getattr(args, 'stat_memory', False)) and _PROTO_STATS_MEMORY is not None:
        hard_statistical_replay(args, model, args.num_labeled_classes + args.way)

    return acc


def calibrate_novel_prototype(proto, old_protos, alpha=0.9, topk=8,
                              temperature=0.1, preserve_norm=True):
    """TEEN-style, training-free correction for a noisy few-shot class mean.

    Only prototype statistics are transferred; the encoder and all old prototypes
    remain untouched.  Keeping the original norm is important because this model's
    classifier is not a strictly normalized cosine head.
    """
    if old_protos is None or old_protos.numel() == 0 or alpha >= 1.0:
        return proto
    with torch.no_grad():
        p = proto.detach()
        old = old_protos.detach()
        similarity = F.normalize(p.unsqueeze(0), dim=-1) @ F.normalize(old, dim=-1).t()
        k = old.shape[0] if int(topk) <= 0 else max(1, min(int(topk), old.shape[0]))
        values, indices = similarity.squeeze(0).topk(k)
        weights = F.softmax(values / max(float(temperature), 1e-6), dim=0)
        transferred = (weights.unsqueeze(1) * old[indices]).sum(dim=0)
        calibrated = float(alpha) * p + (1.0 - float(alpha)) * transferred
        calibrated = F.normalize(calibrated, dim=0)
        if preserve_norm:
            calibrated = calibrated * p.norm().clamp_min(1e-8)
    return calibrated

def optimize_prototype_compactness(proto, features, lr=0.05, steps=30,
                                    base_protos=None, base_margin=0.2,
                                    novel_protos=None, novel_margin=0.0, novel_weight=1.0):
    """
    使用 MSE Loss 强行压缩，并加入与 base 原型的余弦距离下限约束，
    防止新类原型被 MSE 压缩到接近任意已有原型（避免 session3 的 known acc 坍塌）。
    Z-b: 额外支持 novel_protos (同 session 内已更新的新类原型) 的 repel 约束，
    使 session 内 novel-novel 保持角度分离。
    """
    target_proto = proto.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([target_proto], lr=lr, momentum=0.9)

    with torch.enable_grad():
        for _ in range(steps):
            loss_compact = F.mse_loss(target_proto.expand_as(features), features.detach())

            loss_repel = torch.tensor(0.0, device=target_proto.device)
            if base_protos is not None and base_protos.numel() > 0:
                # cosine similarity to every base proto
                p_norm = F.normalize(target_proto.unsqueeze(0), dim=-1)
                b_norm = F.normalize(base_protos.detach(), dim=-1)
                cos_sim = (p_norm @ b_norm.t()).squeeze(0)    # [num_base]
                # 若最大相似度 > 1 - base_margin，则施加推离惩罚
                thresh = 1.0 - base_margin
                violation = torch.clamp(cos_sim - thresh, min=0.0)
                loss_repel = violation.pow(2).sum()

            # Z-b: novel-novel repel within the same session
            loss_novel = torch.tensor(0.0, device=target_proto.device)
            if novel_protos is not None and novel_protos.numel() > 0 and novel_margin > 0.0:
                p_norm2 = F.normalize(target_proto.unsqueeze(0), dim=-1)
                n_norm = F.normalize(novel_protos.detach(), dim=-1)
                cos_sim_n = (p_norm2 @ n_norm.t()).squeeze(0)
                thresh_n = 1.0 - novel_margin
                violation_n = torch.clamp(cos_sim_n - thresh_n, min=0.0)
                loss_novel = violation_n.pow(2).sum()

            loss = loss_compact + 0.5 * loss_repel + novel_weight * loss_novel
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return target_proto.detach()


def refine_session_prototypes(args, model, data, labels, session=None):
    """
    Session-level joint refinement for seen prototypes.

    The existing clustering path only rewrites novel-class prototypes. This helper
    optionally refines all prototypes that appear in the current evaluation batch,
    while anchoring base prototypes to their pre-refine positions.
    """
    if data is None or labels is None or len(data) == 0 or len(labels) == 0:
        return 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_mode = model.mode
    features_list = []
    lfc = _get_lfc(device, k_ratio=0.4)

    try:
        with torch.no_grad():
            model.mode = 'incre'
            for x in data:
                feat = model.hgnn_encode(x.to(device))
                feat, _ = lfc(feat)
                features_list.append(feat.squeeze())
    finally:
        model.mode = original_mode

    if len(features_list) == 0:
        return 0

    features = torch.stack(features_list).to(device)
    label_tensor = torch.as_tensor(labels, device=device, dtype=torch.long)
    seen_classes = sorted(set(int(v) for v in label_tensor.detach().cpu().tolist()))
    if len(seen_classes) == 0:
        return 0

    base_count = min(int(args.num_labeled_classes), model.fc.weight.size(0))
    base_snapshot = model.fc.weight[:base_count].detach().clone().to(device)
    novel_classes = [cls for cls in seen_classes if cls >= base_count]

    # If only base classes are present, there is nothing novel to adapt around.
    if len(novel_classes) == 0:
        return 0

    proto_ids = list(range(base_count)) + novel_classes
    proto_lookup = {cls: idx for idx, cls in enumerate(proto_ids)}
    proto = model.fc.weight[proto_ids].detach().clone().to(device)
    proto.requires_grad_(True)
    optimizer = torch.optim.SGD([proto], lr=float(getattr(args, 'joint_proto_refine_lr', 0.01)), momentum=0.9)

    class_means = {}
    for cls in seen_classes:
        cls_mask = label_tensor == cls
        if cls_mask.any():
            class_means[cls] = features[cls_mask].mean(dim=0).detach()

    refine_steps = int(getattr(args, 'joint_proto_refine_steps', 10))
    anchor_weight = float(getattr(args, 'joint_proto_refine_anchor_weight', 0.1))
    sep_weight = float(getattr(args, 'joint_proto_refine_sep_weight', 0.1))
    sep_margin = float(getattr(args, 'joint_proto_refine_sep_margin', 0.15))

    eye_mask = None
    with torch.enable_grad():
        for _ in range(refine_steps):
            loss_compact = torch.tensor(0.0, device=device)
            for cls, cls_mean in class_means.items():
                loss_compact = loss_compact + F.mse_loss(proto[proto_lookup[cls]], cls_mean)

            loss_anchor = F.mse_loss(proto[:base_count], base_snapshot)

            loss_sep = torch.tensor(0.0, device=device)
            if proto.size(0) > 1:
                proto_norm = F.normalize(proto, dim=-1)
                cos_sim = proto_norm @ proto_norm.t()
                if eye_mask is None or eye_mask.size(0) != cos_sim.size(0):
                    eye_mask = torch.eye(cos_sim.size(0), dtype=torch.bool, device=device)
                off_diag = cos_sim.masked_select(~eye_mask)
                if off_diag.numel() > 0:
                    loss_sep = torch.relu(off_diag - sep_margin).pow(2).mean()

            loss = loss_compact + anchor_weight * loss_anchor + sep_weight * loss_sep
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.fc.weight.data[:base_count] = proto[:base_count].detach()
        for offset, cls in enumerate(novel_classes):
            model.fc.weight.data[cls] = proto[base_count + offset].detach()

    print(f'  [PROTO-REFINE] session={session} seen={len(seen_classes)} novel={len(novel_classes)} '
          f'steps={refine_steps} lr={float(getattr(args, "joint_proto_refine_lr", 0.01)):.4f}')
    return len(seen_classes)


def adapt_old_prototypes_after_expansion(args, model, data, labels, session=None):
    """
    Adapt old prototypes after new-class prototype expansion.

    Only prototypes in [0, args.num_labeled_classes) are updated. The objective
    keeps them anchored, pulls classes observed in current known data toward
    their feature means, and separates every old prototype from the newly added
    prototypes. A guarded call restores the snapshot if current known accuracy
    drops after adaptation.
    """
    old_count = int(args.num_labeled_classes)
    new_end = min(old_count + int(args.way), model.fc.weight.size(0))
    if old_count <= 0 or new_end <= old_count:
        return 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    old_snapshot = model.fc.weight[:old_count].detach().clone().to(device)
    new_snapshot = model.fc.weight[old_count:new_end].detach().clone().to(device)
    if new_snapshot.numel() == 0:
        return 0

    valid_pairs = []
    if data is not None and labels is not None:
        valid_pairs = [(x, int(y)) for x, y in zip(data, labels) if int(y) < old_count]

    before_acc = None
    if getattr(args, 'old_proto_adapt_guard', True) and valid_pairs:
        before_acc, _ = known_test(args, model, [x for x, _ in valid_pairs], [y for _, y in valid_pairs])

    class_means = {}
    if valid_pairs:
        original_mode = model.mode
        feats_by_class = {}
        try:
            with torch.no_grad():
                model.mode = 'incre'
                for x, y in valid_pairs:
                    feat = model.encode(x.to(device)).squeeze()
                    feats_by_class.setdefault(y, []).append(feat)
        finally:
            model.mode = original_mode

        for cls, feats in feats_by_class.items():
            if len(feats) > 0:
                class_means[cls] = torch.stack(feats).mean(dim=0).detach()

    old_proto = old_snapshot.clone().requires_grad_(True)
    optimizer = torch.optim.SGD(
        [old_proto],
        lr=float(getattr(args, 'old_proto_adapt_lr', 0.01)),
        momentum=0.9,
    )

    steps = int(getattr(args, 'old_proto_adapt_steps', 10))
    anchor_weight = float(getattr(args, 'old_proto_adapt_anchor_weight', 1.0))
    compact_weight = float(getattr(args, 'old_proto_adapt_compact_weight', 1.0))
    sep_weight = float(getattr(args, 'old_proto_adapt_sep_weight', 0.1))
    sep_margin = float(getattr(args, 'old_proto_adapt_sep_margin', 0.35))

    with torch.enable_grad():
        for _ in range(steps):
            loss_anchor = F.mse_loss(old_proto, old_snapshot)

            loss_compact = torch.tensor(0.0, device=device)
            for cls, cls_mean in class_means.items():
                loss_compact = loss_compact + F.mse_loss(old_proto[cls], cls_mean)
            if len(class_means) > 0:
                loss_compact = loss_compact / len(class_means)

            old_norm = F.normalize(old_proto, dim=-1)
            new_norm = F.normalize(new_snapshot.detach(), dim=-1)
            cos_old_new = old_norm @ new_norm.t()
            loss_sep = torch.relu(cos_old_new - sep_margin).pow(2).mean()

            loss = anchor_weight * loss_anchor + compact_weight * loss_compact + sep_weight * loss_sep
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.fc.weight.data[:old_count] = old_proto.detach()

    after_acc = None
    restored = False
    if before_acc is not None:
        after_acc, _ = known_test(args, model, [x for x, _ in valid_pairs], [y for _, y in valid_pairs])
        if after_acc + 1e-8 < before_acc:
            with torch.no_grad():
                model.fc.weight.data[:old_count] = old_snapshot
            restored = True

    msg = (f'  [OLD-PROTO-ADAPT] session={session} old={old_count} new={new_end - old_count} '
           f'seen_old={len(class_means)} steps={steps} lr={float(getattr(args, "old_proto_adapt_lr", 0.01)):.4f}')
    if before_acc is not None:
        msg += f' known_acc={before_acc:.4f}->{after_acc:.4f}'
    if restored:
        msg += ' restored'
    print(msg)
    return len(class_means)
# def debug_cluster(args, model, data, labels, session=None):
#     """改进的特征聚类函数（带时序约束）"""
#     with torch.no_grad():
#         features = torch.stack([model.hgnn_encode(x).squeeze() for x in data])  # [N,512,H,W]
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         features,_ = LocalFeatureCluster(k_ratio=0.4)(features)
#         features = features.to(device)
#         kmeans = KMeans(n_clusters=args.num_unlabeled_classes, n_init=20).fit(features.cpu().numpy())
        
#         # 原型更新
#         y = kmeans.labels_
#         acc, map = cluster_acc(args, np.array(labels), y)
        
#         updated = 0
#         for cluster_id in np.unique(y):
#             if cluster_id in map:
#                 true_label = map[cluster_id]
#                 if true_label >= args.num_labeled_classes:
#                     indices = np.where(y == cluster_id)[0]
#                     if len(indices) > 0:
#                         new_proto =features[indices].mean(dim=0).to('cuda')  # 使用压缩后的特征
#                         model.fc.weight.data[true_label] = new_proto
#                         updated += 1
    
#     return acc

_LFC_CACHE = None
_STRUCTURE_DISCOVERY_MODEL = None

def _get_structure_discovery_model(args, device):
    global _STRUCTURE_DISCOVERY_MODEL
    path = str(getattr(args, 'structure_discovery_checkpoint', '') or '')
    if not path or float(getattr(args, 'structure_discovery_weight', 0.0)) <= 0:
        return None
    if _STRUCTURE_DISCOVERY_MODEL is None:
        structure_model = MYNET(args, mode='extract_feature').to(device)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        structure_model.load_state_dict(payload.get('params', payload), strict=False)
        structure_model.eval()
        for parameter in structure_model.parameters():
            parameter.requires_grad_(False)
        _STRUCTURE_DISCOVERY_MODEL = structure_model
        print(f'[STRUCTURE-DISCOVERY] loaded frozen DFSB encoder: {path}')
    return _STRUCTURE_DISCOVERY_MODEL
def _get_lfc(device, k_ratio=0.4):
    global _LFC_CACHE
    if _LFC_CACHE is None:
        _LFC_CACHE = LocalFeatureCluster(k_ratio=k_ratio).to(device)
    return _LFC_CACHE


def estimate_base_feature_variance(args, model, trainset):
    """Estimate classwise diagonal variance once from labeled base embeddings."""
    # This diagnostic pass must not change the subsequent paired stream.
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    # Some encoders keep non-BatchNorm running buffers in custom forward paths.
    # Make the statistics pass strictly read-only for every parameter/buffer.
    model_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=False,
        num_workers=args.dataloader.num_workers, pin_memory=True)
    sums = None; sq_sums = None
    counts = torch.zeros(args.num_base, dtype=torch.float64)
    mode = model.mode
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for data, label in loader:
            data = data.cuda(non_blocking=True)
            model.mode = 'incre'
            z = F.normalize(model.encode(data), dim=-1).detach().cpu().double()
            if sums is None:
                sums = torch.zeros(args.num_base, z.size(1), dtype=torch.float64)
                sq_sums = torch.zeros_like(sums)
            for cls in label.unique():
                c = int(cls)
                mask = label == cls
                sums[c] += z[mask].sum(0)
                sq_sums[c] += z[mask].pow(2).sum(0)
                counts[c] += int(mask.sum())
    model.mode = mode
    model.load_state_dict(model_state, strict=True)
    model.train(was_training)
    mean = sums / counts.clamp_min(1)[:, None]
    variance = sq_sums / counts.clamp_min(1)[:, None] - mean.pow(2)
    variance = variance.float().clamp(1e-6, 0.05)
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.random.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    print(f'[STAT-MEM] estimated base diagonal variance: mean={variance.mean():.6g} '
          f'total={variance.sum(1).mean():.6g}')
    return variance

_INCREMENTAL_OSR_GATE_THRESHOLDS = {}
_TREE_ROUTER_CACHE = {}
_RADIUS_CACHE = {}
_QUANTILE_ROUTER_THRESHOLDS = {}
_SINKHORN_CLASS_BIASES = {}


def _incremental_cosine_bank_logits(args, model, data, test_class):
    """Classifier-consistent logits used by the label-free calibration prepass."""
    query = model.encode(data)
    proto = model.fc.weight[:test_class, :].detach()
    logits = F.cosine_similarity(query.unsqueeze(1), proto, dim=-1)
    if getattr(args, 'novel_bank_classifier', False):
        for cls_id, bank in _NOVEL_SUPPORT_BANK.items():
            if args.num_base <= cls_id < test_class and bank.numel() > 0:
                bank = bank.to(query.device)
                scores = F.cosine_similarity(query.unsqueeze(1), bank.unsqueeze(0), dim=-1)
                topk = min(max(1, int(getattr(args, 'novel_bank_topk', 1))), scores.size(1))
                logits[:, cls_id] = scores.topk(topk, dim=1).values.mean(dim=1)
    return logits


def test(args, model, testloader,  session):
    test_class = args.num_base + session * args.way
    model = model.eval()
    num_batch=0
    va=0.0
    sup_emb, novel_ids = None, None
    if (getattr(args, 'incremental_sinkhorn_balance', False) and session > 0 and
            int(session) not in _SINKHORN_CLASS_BIASES):
        if str(getattr(args, 'incremental_metric', 'cosine')) != 'cosine':
            raise ValueError('incremental_sinkhorn_balance currently requires cosine metric')
        all_logits = []
        with torch.no_grad():
            for balance_batch in testloader:
                # Intentionally ignore balance_batch[1]: this calibration is label-free.
                balance_data = balance_batch[0].cuda()
                all_logits.append(_incremental_cosine_bank_logits(
                    args, model, balance_data, test_class).float().cpu())
        all_logits = torch.cat(all_logits, dim=0)
        temperature = max(float(getattr(args, 'incremental_sinkhorn_temperature', 0.05)), 1e-4)
        iterations = max(1, int(getattr(args, 'incremental_sinkhorn_iterations', 100)))
        scope = str(getattr(args, 'incremental_sinkhorn_scope', 'class'))
        if scope == 'group':
            base_end = int(args.num_base)
            # Max pooling matches the eventual nearest-prototype decision and
            # avoids the 69-way log-sum-exp multiplicity bias of the base group.
            base_score = all_logits[:, :base_end].max(dim=1).values
            novel_score = all_logits[:, base_end:].max(dim=1).values
            target_fraction = float(test_class - base_end) / float(test_class)
            delta = torch.tensor(0.0, dtype=all_logits.dtype)
            for _ in range(iterations):
                probability = torch.sigmoid((novel_score + delta - base_score) / temperature)
                current = probability.mean().clamp(1e-6, 1.0 - 1e-6)
                delta += temperature * (torch.logit(torch.tensor(target_fraction)) - torch.logit(current))
            bias = torch.zeros(test_class, dtype=all_logits.dtype)
            bias[base_end:] = delta
            final_fraction = torch.sigmoid((novel_score + delta - base_score) / temperature).mean()
            count_summary = f'novel_fraction={final_fraction:.6f} target={target_fraction:.6f}'
        else:
            bias = torch.zeros(test_class, dtype=all_logits.dtype)
            target_count = float(all_logits.size(0)) / float(test_class)
            for _ in range(iterations):
                scaled = (all_logits + bias.unsqueeze(0)) / temperature
                assignment = torch.softmax(scaled, dim=1)
                column_count = assignment.sum(dim=0).clamp_min(1e-6)
                bias += temperature * torch.log(torch.full_like(column_count, target_count) / column_count)
            bias -= bias.mean()
            final_count = torch.softmax((all_logits + bias.unsqueeze(0)) / temperature, dim=1).sum(0)
            count_summary = f'count_min={final_count.min():.2f} count_max={final_count.max():.2f}'
        _SINKHORN_CLASS_BIASES[int(session)] = bias
        print(f'[SINKHORN-BALANCE] session={session} samples={all_logits.size(0)} classes={test_class} '
              f'scope={scope} temperature={temperature:.6f} iterations={iterations} {count_summary} '
              f'bias_min={bias.min():.6f} bias_max={bias.max():.6f}')
    if (getattr(args, 'incremental_quantile_group_gate', False) and session > 0 and
            int(session) not in _QUANTILE_ROUTER_THRESHOLDS):
        active_banks = [bank.cuda() for cls_id, bank in sorted(_NOVEL_SUPPORT_BANK.items())
                        if args.num_base <= cls_id < test_class and bank.numel() > 0]
        if active_banks:
            base_proto = F.normalize(model.fc.weight[:int(args.num_base)].detach(), dim=1)
            novelty_scores = []
            with torch.no_grad():
                for quantile_batch in testloader:
                    quantile_data = quantile_batch[0].cuda()
                    quantile_query = F.normalize(model.encode(quantile_data), dim=1)
                    base_max = (quantile_query @ base_proto.t()).max(dim=1).values
                    per_class_scores = []
                    for bank in active_banks:
                        support_scores = quantile_query @ F.normalize(bank, dim=1).t()
                        q_topk = min(max(1, int(getattr(args, 'incremental_quantile_support_topk', 1))),
                                     support_scores.size(1))
                        per_class_scores.append(support_scores.topk(q_topk, dim=1).values.mean(dim=1))
                    class_scores = torch.stack(per_class_scores, dim=1)
                    novel_top = class_scores.topk(min(2, class_scores.size(1)), dim=1).values
                    novel_max = novel_top[:, 0]
                    novel_gap = novel_top[:, 0] - novel_top[:, -1]
                    score_mode = str(getattr(args, 'incremental_quantile_score', 'support_margin'))
                    novelty = (novel_max if score_mode == 'novel_max' else
                               novel_gap if score_mode == 'novel_gap' else
                               novel_max - base_max + novel_gap if score_mode == 'margin_gap' else
                               novel_max - base_max)
                    novelty_scores.append(novelty)
            novelty_scores = torch.cat(novelty_scores)
            expected_fraction = (float(session * args.way) /
                                 float(args.num_base + session * args.way))
            threshold = torch.quantile(novelty_scores, 1.0 - expected_fraction).item()
            _QUANTILE_ROUTER_THRESHOLDS[int(session)] = threshold
            print(f'[QUANTILE-GATE] session={session} samples={len(novelty_scores)} '
                  f'novel_fraction={expected_fraction:.6f} threshold={threshold:.6f}')
    if (getattr(args, 'incremental_osr_group_gate', False) and session > 0 and
            int(session) not in _INCREMENTAL_OSR_GATE_THRESHOLDS):
        gate_margins = []
        with torch.no_grad():
            for gate_batch in testloader:
                gate_data = gate_batch[0].cuda()
                gate_feature = model.hgnn_encode(gate_data)
                if gate_feature.dim() > 2:
                    gate_feature = F.adaptive_avg_pool2d(gate_feature, 1).flatten(1)
                gate_base = int(args.num_base)
                gate_proto = model.fc.weight[:gate_base].detach()
                gate_ids = torch.arange(gate_base, device=gate_data.device)
                gate_scores = model.cls_classifier.incre_forward(
                    gate_feature, gate_proto, gate_ids, osr_noise_std=0.0).squeeze()
                if gate_scores.dim() == 1:
                    gate_scores = gate_scores.unsqueeze(0)
                gate_pos = gate_scores[:, :gate_base]
                gate_neg = gate_scores[:, gate_base:gate_base * 2]
                gate_pred = gate_pos.argmax(dim=1)
                gate_margins.append(gate_pos.gather(1, gate_pred[:, None]).squeeze(1) -
                                    gate_neg.gather(1, gate_pred[:, None]).squeeze(1))
        all_gate_margins = torch.cat(gate_margins)
        _INCREMENTAL_OSR_GATE_THRESHOLDS[int(session)] = _adaptive_margin_threshold(all_gate_margins)
        print(f'[INC-OSR-GATE] session={session} samples={len(all_gate_margins)} '
              f'threshold={_INCREMENTAL_OSR_GATE_THRESHOLDS[int(session)]:.6f}')
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            model.mode = 'incre'
            query = model.encode(data)  # [B, 512] — consistent with prototype generation
            # print(f"Original query shape: {query.shape}")
            proto = model.fc.weight[:test_class, :].detach()
            if getattr(args, 'feature_centering', False):
                center = model.fc.weight[:args.num_base, :].detach().mean(dim=0, keepdim=True)
                query = query - center
                proto = proto - center
            metric = str(getattr(args, 'incremental_metric', 'cosine'))
            if getattr(args, 'use_pan_incremental', False) and session > 0:
                novel_ids = sorted(cls for cls in _NOVEL_SUPPORT_BANK
                                   if args.num_base <= cls < test_class)
                shot = int(args.episode.episode_shot)
                if novel_ids and all(_NOVEL_SUPPORT_BANK[cls].shape[0] >= shot for cls in novel_ids):
                    # PAN expects shot-major ordering: c0_s0,c1_s0,...,c0_s1,c1_s1,...
                    support_embeddings = torch.stack([
                        _NOVEL_SUPPORT_BANK[cls][s].to(query.device)
                        for s in range(shot) for cls in novel_ids
                    ])
                    support_proto = proto.unsqueeze(0).unsqueeze(0)
                    query_episode = query.unsqueeze(0).unsqueeze(2)
                    logits, _, _ = model._forward(
                        support_proto, query_episode, pqa=True,
                        sup_emb=support_embeddings, novel_ids=novel_ids)
                else:
                    logits = (-torch.cdist(query, proto).pow(2) if metric == 'euclidean'
                              else query @ proto.t() if metric == 'dot'
                              else F.cosine_similarity(query.unsqueeze(1), proto, dim=-1))
            else:
                logits = (-torch.cdist(query, proto).pow(2) if metric == 'euclidean'
                          else query @ proto.t() if metric == 'dot'
                          else F.cosine_similarity(query.unsqueeze(1), proto, dim=-1))
            if getattr(args, 'novel_bank_classifier', False) and not getattr(args, 'use_pan_incremental', False):
                for cls_id, bank in _NOVEL_SUPPORT_BANK.items():
                    if args.num_base <= cls_id < test_class and bank.numel() > 0:
                        bank = bank.to(query.device)
                        bank_scores = (-torch.cdist(query, bank).pow(2) if metric == 'euclidean'
                                       else query @ bank.t() if metric == 'dot'
                                       else F.cosine_similarity(query.unsqueeze(1), bank.unsqueeze(0), dim=-1))
                        bank_topk = min(max(1, int(getattr(args, 'novel_bank_topk', 1))),
                                        bank_scores.size(1))
                        top_values = bank_scores.topk(bank_topk, dim=1).values
                        bank_temp = float(getattr(args, 'novel_bank_temperature', 0.0))
                        if bank_temp > 0.0:
                            weights = F.softmax(top_values / bank_temp, dim=1)
                            bank_value = (weights * top_values).sum(dim=1)
                        else:
                            bank_value = top_values.mean(dim=1)
                        bank_blend = float(getattr(args, 'novel_bank_blend', 1.0))
                        bank_blend = min(max(bank_blend, 0.0), 1.0)
                        mean_value = (query @ proto[cls_id].to(query).unsqueeze(1)).squeeze(1) \
                            if metric == 'dot' else F.cosine_similarity(
                                query, proto[cls_id].to(query).unsqueeze(0), dim=1)
                        logits[:, cls_id] = bank_blend * bank_value + (1.0 - bank_blend) * mean_value
            if (getattr(args, 'incremental_sinkhorn_balance', False) and
                    int(session) in _SINKHORN_CLASS_BIASES):
                logits = logits + _SINKHORN_CLASS_BIASES[int(session)].to(logits).unsqueeze(0)
            radius_power = float(getattr(args, 'incremental_radius_power', 0.0))
            if radius_power > 0.0 and metric == 'cosine':
                radius_key = (int(session), int(test_class))
                if radius_key not in _RADIUS_CACHE:
                    geometry_path = str(getattr(args, 'base_geometry_path', ''))
                    geometry = torch.load(geometry_path, map_location='cpu', weights_only=True)
                    radii = torch.empty(test_class, device=query.device)
                    for cls_id in range(min(int(args.num_base), test_class)):
                        pool = geometry['class_features'][cls_id].to(query.device)
                        similarity = (F.normalize(pool, dim=1) @ F.normalize(proto[cls_id], dim=0))
                        radii[cls_id] = (1.0 - similarity).mean().clamp_min(1e-3)
                    base_median = radii[:min(int(args.num_base), test_class)].median()
                    for cls_id in range(int(args.num_base), test_class):
                        bank = _NOVEL_SUPPORT_BANK.get(cls_id)
                        if bank is None or bank.numel() == 0:
                            radii[cls_id] = base_median
                        else:
                            similarity = (F.normalize(bank.to(query.device), dim=1) @
                                          F.normalize(proto[cls_id], dim=0))
                            empirical = (1.0 - similarity).mean().clamp_min(1e-3)
                            radii[cls_id] = 0.5 * empirical + 0.5 * base_median
                    _RADIUS_CACHE[radius_key] = radii
                radii = _RADIUS_CACHE[radius_key]
                logits = -(1.0 - logits) / radii.pow(radius_power).unsqueeze(0)
            novel_bias = float(getattr(args, 'incremental_novel_logit_bias', 0.0))
            if novel_bias != 0.0 and logits.size(1) > int(args.num_base):
                logits[:, int(args.num_base):] -= novel_bias
            novel_scale = float(getattr(args, 'incremental_novel_logit_scale', 1.0))
            if novel_scale != 1.0 and logits.size(1) > int(args.num_base):
                logits[:, int(args.num_base):] *= novel_scale
            base_scale = float(getattr(args, 'incremental_base_logit_scale', 1.0))
            if base_scale != 1.0 and logits.size(1) > int(args.num_base):
                logits[:, :int(args.num_base)] *= base_scale
            hubness_weight = float(getattr(args, 'incremental_proto_hubness_weight', 0.0))
            if hubness_weight != 0.0 and metric == 'cosine' and proto.size(0) > 1:
                # A few-shot mean can become a cosine-space hub and attract many
                # unrelated queries.  Estimate that bias solely from the current
                # classifier geometry; unlike a global novel bias this correction
                # is class-specific and does not require labels or another encoder.
                proto_unit = F.normalize(proto, dim=1)
                proto_affinity = proto_unit @ proto_unit.t()
                proto_affinity.fill_diagonal_(-float('inf'))
                hub_k = min(max(1, int(getattr(args, 'incremental_proto_hubness_k', 8))),
                            proto.size(0) - 1)
                hubness = proto_affinity.topk(hub_k, dim=1).values.mean(dim=1)
                hubness_scope = str(getattr(args, 'incremental_proto_hubness_scope', 'all'))
                hubness_mask = torch.ones_like(hubness)
                if hubness_scope == 'base':
                    hubness_mask[int(args.num_base):] = 0.0
                elif hubness_scope == 'novel':
                    hubness_mask[:int(args.num_base)] = 0.0
                logits = logits - hubness_weight * hubness.unsqueeze(0) * hubness_mask.unsqueeze(0)
            router_path = str(getattr(args, 'incremental_group_router_path', ''))
            if router_path and logits.size(1) > int(args.num_base):
                router = torch.load(router_path, map_location=logits.device, weights_only=True)
                base_end = int(args.num_base)
                base_top = logits[:, :base_end].topk(2, dim=1).values
                novel_top = logits[:, base_end:].topk(2, dim=1).values
                router_x = torch.stack([
                    base_top[:, 0], novel_top[:, 0], base_top[:, 0] - base_top[:, 1],
                    novel_top[:, 0] - novel_top[:, 1], novel_top[:, 0] - base_top[:, 0]], dim=1)
                router_score = (router_x @ router['coef'].to(router_x) + float(router['intercept']) +
                                float(getattr(args, 'incremental_group_router_offset', 0.0)))
                router_soft_scale = float(getattr(args, 'incremental_group_router_soft_scale', 0.0))
                if router_soft_scale > 0.0:
                    logits[:, base_end:] += router_soft_scale * router_score.unsqueeze(1)
                else:
                    choose_novel = router_score > 0
                    logits[choose_novel, :base_end] = -float('inf')
                    logits[~choose_novel, base_end:] = -float('inf')
            tree_router_path = str(getattr(args, 'incremental_tree_router_path', ''))
            if tree_router_path and logits.size(1) > int(args.num_base):
                if tree_router_path not in _TREE_ROUTER_CACHE:
                    _TREE_ROUTER_CACHE[tree_router_path] = joblib.load(tree_router_path)
                base_end = int(args.num_base)
                base_top = logits[:, :base_end].topk(2, dim=1).values
                novel_top = logits[:, base_end:].topk(2, dim=1).values
                tree_x = torch.stack([base_top[:, 0], novel_top[:, 0],
                                      base_top[:, 0] - base_top[:, 1],
                                      novel_top[:, 0] - novel_top[:, 1],
                                      novel_top[:, 0] - base_top[:, 0]], dim=1)
                tree_model = _TREE_ROUTER_CACHE[tree_router_path]
                if int(getattr(tree_model, 'n_features_in_', 5)) >= 7:
                    active_banks = [bank.to(query.device) for cls_id, bank in _NOVEL_SUPPORT_BANK.items()
                                    if args.num_base <= cls_id < test_class and bank.numel() > 0]
                    if active_banks:
                        support = F.normalize(torch.cat(active_banks, dim=0), dim=1)
                        support_score = F.normalize(query, dim=1) @ support.t()
                        support_top = support_score.topk(min(2, support_score.size(1)), dim=1).values
                        tree_x = torch.cat([tree_x, support_top[:, :1],
                                            support_top.mean(dim=1, keepdim=True)], dim=1)
                probability = tree_model.predict_proba(
                    tree_x.detach().cpu().numpy())[:, 1]
                probability = torch.from_numpy(probability).to(logits).clamp(1e-5, 1-1e-5)
                tree_scale = float(getattr(args, 'incremental_tree_router_soft_scale', 0.0))
                if tree_scale > 0.0:
                    logits[:, base_end:] += tree_scale * torch.logit(probability).unsqueeze(1)
                else:
                    choose_novel = probability > 0.5
                    logits[choose_novel, :base_end] = -float('inf')
                    logits[~choose_novel, base_end:] = -float('inf')
            if getattr(args, 'oracle_eval_group_gate', False) and logits.size(1) > int(args.num_base):
                base_end = int(args.num_base)
                diagnostic_novel = test_label >= base_end
                logits[diagnostic_novel, :base_end] = -float('inf')
                logits[~diagnostic_novel, base_end:] = -float('inf')
            if (getattr(args, 'incremental_quantile_group_gate', False) and
                    int(session) in _QUANTILE_ROUTER_THRESHOLDS and logits.size(1) > int(args.num_base)):
                base_end = int(args.num_base)
                active_banks = [bank.to(query.device) for cls_id, bank in sorted(_NOVEL_SUPPORT_BANK.items())
                                if args.num_base <= cls_id < test_class and bank.numel() > 0]
                base_max = (F.normalize(query, dim=1) @
                            F.normalize(proto[:base_end], dim=1).t()).max(dim=1).values
                per_class_scores = []
                for bank in active_banks:
                    support_scores = F.normalize(query, dim=1) @ F.normalize(bank, dim=1).t()
                    q_topk = min(max(1, int(getattr(args, 'incremental_quantile_support_topk', 1))),
                                 support_scores.size(1))
                    per_class_scores.append(support_scores.topk(q_topk, dim=1).values.mean(dim=1))
                class_scores = torch.stack(per_class_scores, dim=1)
                novel_top = class_scores.topk(min(2, class_scores.size(1)), dim=1).values
                novel_max = novel_top[:, 0]
                novel_gap = novel_top[:, 0] - novel_top[:, -1]
                score_mode = str(getattr(args, 'incremental_quantile_score', 'support_margin'))
                novelty = (novel_max if score_mode == 'novel_max' else
                           novel_gap if score_mode == 'novel_gap' else
                           novel_max - base_max + novel_gap if score_mode == 'margin_gap' else
                           novel_max - base_max)
                choose_novel = novelty >= float(_QUANTILE_ROUTER_THRESHOLDS[int(session)])
                logits[choose_novel, :base_end] = -float('inf')
                logits[~choose_novel, base_end:] = -float('inf')
            if (getattr(args, 'incremental_osr_group_gate', False) and session > 0 and
                    logits.size(1) > int(args.num_base)):
                # Use the checkpointed open-set head only as a base/novel router.
                # Class prototypes and within-group classification stay in the
                # current model.encode space.
                osr_feature = model.hgnn_encode(data)
                if osr_feature.dim() > 2:
                    osr_feature = F.adaptive_avg_pool2d(osr_feature, 1).flatten(1)
                base_end = int(args.num_base)
                base_proto = model.fc.weight[:base_end].detach()
                base_ids = torch.arange(base_end, device=query.device)
                osr_scores = model.cls_classifier.incre_forward(
                    osr_feature, base_proto, base_ids, osr_noise_std=0.0)
                osr_scores = osr_scores.squeeze()
                if osr_scores.dim() == 1:
                    osr_scores = osr_scores.unsqueeze(0)
                positive = osr_scores[:, :base_end]
                negative = osr_scores[:, base_end:base_end * 2]
                predicted_base = positive.argmax(dim=1)
                pos_score = positive.gather(1, predicted_base[:, None]).squeeze(1)
                neg_score = negative.gather(1, predicted_base[:, None]).squeeze(1)
                osr_margin = pos_score - neg_score
                threshold = float(_INCREMENTAL_OSR_GATE_THRESHOLDS[int(session)])
                choose_novel = osr_margin <= threshold
                logits[choose_novel, :base_end] = -float('inf')
                logits[~choose_novel, base_end:] = -float('inf')
            if (getattr(args, 'incremental_group_margin_gate', False) and
                    logits.size(1) > int(args.num_base) and int(args.num_base) >= 2 and
                    logits.size(1) - int(args.num_base) >= 2):
                base_end = int(args.num_base)
                base_values, base_indices = logits[:, :base_end].topk(2, dim=1)
                novel_values, novel_indices = logits[:, base_end:].topk(2, dim=1)
                base_margin = base_values[:, 0] - base_values[:, 1]
                novel_margin = novel_values[:, 0] - novel_values[:, 1]
                novel_margin = novel_margin + float(getattr(args, 'incremental_group_margin_bias', 0.0))
                choose_novel = novel_margin > base_margin
                routed = torch.full_like(logits, -float('inf'))
                rows = torch.arange(logits.size(0), device=logits.device)
                predicted = torch.where(choose_novel, novel_indices[:, 0] + base_end,
                                        base_indices[:, 0])
                routed[rows, predicted] = logits[rows, predicted]
                logits = routed
            acc = count_acc(logits, test_label)
            num_batch+=1
            va+=acc
    return float(va/num_batch)

#baseline
def known_test(args, model, data, label):
    if len(data) == 0:
        return 0.0, 0.0

    feats = []
    label = torch.tensor(label)
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for i in range(len(data)):
            model.mode = 'incre'
            feat = model.encode(data[i]).to(device)  # [1, 512] — consistent with prototype generation
            feats.append(feat)

    proto = model.fc.weight[:args.num_labeled_classes, :].detach()
    if getattr(args, 'feature_centering', False):
        center = model.fc.weight[:args.num_base, :].detach().mean(dim=0, keepdim=True)
        feats = [feat - center for feat in feats]
        proto = proto - center
    proto = proto.unsqueeze(0).unsqueeze(0)
    feats = torch.stack(feats)
    logits = F.cosine_similarity(feats, proto, dim=-1)
    logits = torch.squeeze(logits)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    target = label.to('cuda')
    if target.dim() == 0:
        target = target.unsqueeze(0)

    acc = count_acc(logits, target)
    preds = torch.argmax(logits, dim=1)
    score = f1_score(label.cpu().numpy(), preds.cpu().numpy(), average='macro')
    return acc, score

def train(args: dict):   
    # ============ base session training ==============
    device = torch.device("cuda" if args.cuda else "cpu")
    model = MYNET(args, mode='encoder')
    model = model.to(device)
    model.apply(weights_init)  # 使用固定种子的初始化
    set_up_datasets(args)
    full_checkpoint_path = str(getattr(args, 'full_checkpoint_path', '') or '')
    if full_checkpoint_path:
        if not os.path.isfile(full_checkpoint_path):
            raise FileNotFoundError(
                f'Local full checkpoint not found: {full_checkpoint_path}. '
                'Offline execution never downloads missing checkpoints.')
        print(f'==> Loading local base-only full checkpoint: {full_checkpoint_path}')
        checkpoint = torch.load(full_checkpoint_path, map_location='cpu', weights_only=True)
        params = checkpoint.get('params', checkpoint)
        if not isinstance(params, dict):
            raise TypeError('Full checkpoint must be a state dict or contain a params state dict')
        incompatible = model.load_state_dict(params, strict=False)
        print(f'    missing_keys={len(incompatible.missing_keys)} '
              f'unexpected_keys={len(incompatible.unexpected_keys)}')
        if 'fc.weight' in params:
            model.cls_classifier.init_representation({'fc.weight': params['fc.weight']})
            # init_representation creates fresh tensors on the checkpoint's CPU
            # device. Restore module/device consistency before CUDA evaluation.
            model.cls_classifier.to(device)
        if getattr(args, 'skip_meta_train', False):
            print('[skip_meta_train=True] bypass meta_train, use local base checkpoint as-is')
        else:
            print('==> Starting Meta Training from local base checkpoint...')
            open_train_val_loader = get_dataloaders(args, 'openmeta')
            meta_train(args, model, open_train_val_loader, eval_loader=None)
    elif args.checkpoint:
        best_model_dir = args.save_dir+'/'+args.checkpoint_name
        #meta-train negative prototype
        params = torch.load(best_model_dir, weights_only=True)['cls_params']
        cls_params = {k: v for k, v in params.items() if 'fc' in k}
        model.cls_classifier.init_representation(cls_params)
        model_dict = model.state_dict()
        model_dict.update(params)
        model.load_state_dict(model_dict)
    else:
        if args.load_base:
            print("=== Mode: Meta Train Only (Skipping Base Train) ===")
            # 加载基础训练的模型
            # 注意：这里的路径要和你 base_train 保存的路径一致
            # 或者是 args.pretrained_model_path，或者是具体的 'base_train_uncertainty.pth'
            
            # 优先尝试加载 args.pretrained_model_path，如果文件不存在则尝试 base_train_uncertainty.pth
            base_model_path = args.pretrained_model_path
            if not os.path.exists(base_model_path):
                base_model_path = os.path.join(args.save_dir, 'base_train_uncertainty.pth')
            
            print(f"==> Loading Base Model from: {base_model_path}")
            
            if os.path.exists(base_model_path):
                checkpoint = torch.load(base_model_path)
                # 加载参数 (使用 strict=False 以防有些不匹配，但通常应该匹配)
                model.load_state_dict(checkpoint['params'], strict=False)
                # 初始化 cls_classifier.weight_base（兼容旧 checkpoint 只有 fc.weight 的情况）
                if 'fc.weight' in checkpoint.get('params', {}):
                    model.cls_classifier.init_representation(
                        {'fc.weight': checkpoint['params']['fc.weight']})
            else:
                raise FileNotFoundError(f"Base model not found at {base_model_path}. Please run with --load_base False first.")
                
        else:
            print("=== Mode: Full Training (Base Train -> Meta Train) ===")
            # 运行基础训练
            best_model_dir = base_train(args, model)
            # 更新 pretrained_model_path，防止 meta_train 加载错误的旧模型
            args.pretrained_model_path = best_model_dir
            
        # 无论是否跳过 Base Train，这里都执行 Meta Train
        if getattr(args, 'skip_meta_train', False):
            print('[skip_meta_train=True] bypass meta_train, use loaded checkpoint as-is')
        else:
            print("==> Starting Meta Training...")
            open_train_val_loader = get_dataloaders(args, 'openmeta')
            meta_train(args, model, open_train_val_loader, eval_loader=None)
        # best_model_dir=base_train(args,model)
        # open_train_val_loader= get_dataloaders(args,'openmeta')
        # meta_train(args, model,open_train_val_loader, eval_loader=None)
    data_dict,result={},{}
    data_dict['train_set'],_=get_pretrain_dataloader(args)
    if getattr(args, 'skip_replace_base_fc', False):
        print('[v7] skip_replace_base_fc=True -> keep meta-trained fc.weight[:num_base]')
    else:
        model = replace_base_fc(args,data_dict['train_set'], model) 

    # Path B: ZCA-whitening on base prototypes to decorrelate while staying in feature-mean space
    if getattr(args, 'orth_base_proto', False):
        import torch as _torch
        _nb = args.num_base
        _s = float(getattr(args, 'orth_strength', 1.0))
        with _torch.no_grad():
            P = model.fc.weight.data[:_nb].detach().clone()           # [80, D]
            norms = P.norm(dim=1, keepdim=True).clamp(min=1e-8)
            Pn = P / norms                                             # row-normalized
            C = Pn @ Pn.t()                                            # [80,80]
            eps = 1e-3
            C = C + eps * _torch.eye(_nb, device=C.device, dtype=C.dtype)
            # ZCA whitening matrix W = C^{-1/2} via eigen-decomposition
            evals, evecs = _torch.linalg.eigh(C)
            evals = evals.clamp(min=eps)
            W = evecs @ _torch.diag(evals.rsqrt()) @ evecs.t()
            P_white = W @ Pn                                           # [80, D]
            # re-normalize rows, then scale back to original norms
            P_white = P_white / P_white.norm(dim=1, keepdim=True).clamp(min=1e-8)
            P_white = P_white * norms
            # blend
            P_final = (1.0 - _s) * P + _s * P_white
            model.fc.weight.data[:_nb] = P_final
            # diagnostic print
            pn = P / norms
            fn = P_final / P_final.norm(dim=1, keepdim=True).clamp(min=1e-8)
            before = (pn @ pn.t())[~_torch.eye(_nb, dtype=_torch.bool, device=pn.device)].mean().item()
            after = (fn @ fn.t())[~_torch.eye(_nb, dtype=_torch.bool, device=fn.device)].mean().item()
            print(f'[v8] orth_base_proto strength={_s:.2f}  base pairwise cos: {before:.4f} -> {after:.4f}')

    # Task 1.1: 结果输出按 opt_version 隔离，避免互相覆盖
    opt_dir = os.path.join(args.save_result, getattr(args, 'opt_version', 'opt_v5'))
    os.makedirs(opt_dir, exist_ok=True)
    tag = getattr(args, 'run_tag', '') or ''
    suffix = f"_{tag}" if tag else ""
    result_path = os.path.join(opt_dir, f'test_result{suffix}.txt')
    _estimated_base_var = None
    if bool(getattr(args, 'stat_memory', False)) and float(getattr(args, 'stat_base_var', -1.0)) < 0:
        _estimated_base_var = estimate_base_feature_variance(args, model, data_dict['train_set'])
    with open(result_path, 'w') as result_file:
        session0_acc_list = []
        session_ka = [[] for _ in range(args.test_times)]
        session_uka = [[] for _ in range(args.test_times)]
        session_auroc_vals = [[] for _ in range(args.test_times)]
        session_f1s = [[] for _ in range(args.test_times)]
        session_inc = [[] for _ in range(args.test_times)]
        session_all = [[] for _ in range(args.test_times)]
        for j in range(0, args.num_session): 
            result['sess{}_ak'.format(j)]=[]
            result['sess{}_au'.format(j)]=[]
            result['sess{}_ar'.format(j)]=[]
            result['sess{}_fs'.format(j)]=[]
            result['sess{}_inc'.format(j)]=[]
            result['sess{}_all'.format(j)]=[]
        # X1: snapshot clean state before test_times loop to prevent fc.weight[80:] pollution
        import copy as _copy
        _clean_state = _copy.deepcopy(model.state_dict())
        _inc_norm0 = float(model.fc.weight.data[args.num_base:].norm().item())
        print(f'[X1] clean_state saved. fc.weight[{args.num_base}:] norm={_inc_norm0:.4f}')
        for i in range(args.test_times):
            global _PROTO_STATS_MEMORY, _NOVEL_SUPPORT_BANK, _LABELED_SUPPORT_PROTOS
            args.current_test = i  # 记录当前测试轮次
            args.num_labeled_classes = args.num_base
            # X1: restore clean state each round to wipe novel-proto pollution
            if bool(getattr(args, 'reset_fc_each_round', True)):
                model.load_state_dict(_clean_state)
            _NOVEL_SUPPORT_BANK = {}
            _LABELED_SUPPORT_PROTOS = {}
            if bool(getattr(args, 'stat_memory', False)):
                base_variance = (_estimated_base_var if _estimated_base_var is not None
                                 else float(getattr(args, 'stat_base_var', 0.0002)))
                _PROTO_STATS_MEMORY = PrototypeStatisticsMemory(
                    model.fc.weight[:args.num_base],
                    base_var=base_variance)
                variance_mean = float(torch.stack(list(_PROTO_STATS_MEMORY.var.values())).mean())
                if variance_mean < float(getattr(args, 'stat_min_variance', 1e-4)):
                    _PROTO_STATS_MEMORY.enabled = False
                    print(f'[STAT-MEM] variance gate disabled replay: mean={variance_mean:.6g} '
                          f'< {float(getattr(args, "stat_min_variance", 1e-4)):.6g}')
            else:
                _PROTO_STATS_MEMORY = None
            reset_session_stats()  # 每轮重置 session-aware 阈值历史
            _INCREMENTAL_OSR_GATE_THRESHOLDS.clear()
            _RADIUS_CACHE.clear()
            _QUANTILE_ROUTER_THRESHOLDS.clear()
            _SINKHORN_CLASS_BIASES.clear()
            print(f"\n=== Base Session Pure Evaluation (Round {i}) ===")
            _, base_testloader = get_testloader(args, 0)  
            base_acc = test(args, model, base_testloader, 0)  
            if (_PROTO_STATS_MEMORY is not None and
                    base_acc < float(getattr(args, 'stat_min_base_acc', 0.7))):
                _PROTO_STATS_MEMORY.enabled = False
                print(f'[STAT-MEM] reliability gate disabled replay: base_acc={base_acc:.4f} '
                      f'< {float(getattr(args, "stat_min_base_acc", 0.7)):.4f}')
            session0_acc_list.append(base_acc)
            # 记录结果（未知类指标设为0）
            result['sess0_ak'].append(base_acc)
            result['sess0_au'].append(0.0)
            result['sess0_fs'].append(0.0)
            result['sess0_inc'].append(0.0)
            result['sess0_all'].append(base_acc)
            # 打印session 0结果
            print(f"Session 0: acc known: {base_acc:.4f}, acc unknown: 0.0000, "
                  f"f1 score: 0.0000, inc acc:0.0000, all acc: {base_acc:.4f}")
            for session in range(args.start_session, args.num_session):  
                print("Inference session: [%d]" % session)
                print(f"test_time: {i}")
                model.mode = args.network.new_mode
                model.eval()
                if getattr(args, 'mixed_openworld_stream', False):
                    _, unlabelled_loader = get_mixed_openworld_dataloader(args, session)
                else:
                    _,unlabelled_loader = get_dataloader(args, session)
                # Labeled few-shot support episode. It is the only source used
                # for novel class IDs; test-stream labels never enter mapping.
                if getattr(args, 'mixed_openworld_stream', False) and \
                        str(getattr(args, 'discovery_encoder', 'hgnn_lfc')) == 'direct':
                    support_targets = np.asarray(getattr(unlabelled_loader.dataset, 'targets', []))
                    support_ids = np.arange(args.num_base + (session - 1) * args.way,
                                            args.num_base + session * args.way)
                    with torch.no_grad():
                        model.mode = 'incre'
                        for cls_id in support_ids:
                            idx = np.flatnonzero(support_targets == cls_id)[:max(1, int(args.n_shots))]
                            if len(idx):
                                emb = []
                                for ii in idx:
                                    sample = unlabelled_loader.dataset[int(ii)][0].unsqueeze(0).cuda()
                                    emb.append(encode_with_deterministic_tta(
                                        model, sample, int(getattr(args, 'encode_tta_views', 1))).squeeze(0))
                                # Preserve every current model.encode support
                                # embedding. The classifier uses their mean, while
                                # the bank classifier can average top-k support
                                # similarities instead of collapsing the 5-shot
                                # episode to one vector.
                                _LABELED_SUPPORT_PROTOS[int(cls_id)] = torch.stack(emb).detach()
                    print(f'  [SUPPORT-ENCODE] session={session} classes={sorted(_LABELED_SUPPORT_PROTOS)} '
                          f'shots={max(1, int(args.n_shots))} source=model.encode')
                #OSR_DETECTION
                unknow_data,unknow_label,know_data,know_label=run_test_fsl(model,args,unlabelled_loader, session=session)
                # Extract per-session AUROC from function attribute
                auroc_list = getattr(run_test_fsl, '_auroc_list', [])
                session_auroc = float(np.mean(auroc_list)) if auroc_list else 0.0
                run_test_fsl._auroc_list = []  # reset for next session

                # ===== OSR 诊断: 分离 known/unknown 判决准确率 =====
                _nl = args.num_labeled_classes
                _all_labels = know_label + unknow_label
                _all_gt_known = [l < _nl for l in _all_labels]
                _all_pred_known = [True]*len(know_label) + [False]*len(unknow_label)
                _tk = sum(1 for gl, pk in zip(_all_gt_known, _all_pred_known) if gl and pk)
                _tu = sum(1 for gl, pk in zip(_all_gt_known, _all_pred_known) if not gl and not pk)
                _fk = sum(1 for gl, pk in zip(_all_gt_known, _all_pred_known) if not gl and pk)
                _fu = sum(1 for gl, pk in zip(_all_gt_known, _all_pred_known) if gl and not pk)
                _total = len(_all_labels)
                _osr_acc = (_tk + _tu) / max(_total, 1)
                _k_recall = _tk / max(_tk + _fu, 1)
                _u_recall = _tu / max(_tu + _fk, 1)
                # 已知类中被漏到 unknown 的有哪些类
                _leaked_known = sorted(set(l for l in unknow_label if l < _nl))
                # 未知类中被错放到 known 的有哪些类
                _missed_unknown = sorted(set(l for l in know_label if l >= _nl))
                print(f'  [OSR-DIAG] total={_total} OSR_acc={_osr_acc:.3f} '
                      f'K_recall={_k_recall:.3f} U_recall={_u_recall:.3f} '
                      f'TK={_tk} FU={_fu} TU={_tu} FK={_fk}')
                if _leaked_known:
                    print(f'  [OSR-DIAG] Leaked known classes → unknown: {_leaked_known}')
                if _missed_unknown:
                    print(f'  [OSR-DIAG] Missed unknown classes → known: {_missed_unknown}')
                # ===== OSR 诊断结束 =====

                #K means
                rank_active = (getattr(args, 'cluster_all_candidates', False) and
                               int(session) >= int(getattr(args, 'discovery_rank_start_session', 1)))
                if rank_active:
                    ranked_topk = int(getattr(args, 'discovery_ranked_topk', 0))
                    rank_score = str(getattr(args, 'discovery_rank_score', 'osr_margin'))
                    if rank_score.startswith('encode_'):
                        all_pairs = list(zip(unknow_data + know_data, unknow_label + know_label))
                        scored_pairs = []
                        encoded_pairs = []
                        model.eval()
                        with torch.no_grad():
                            for sample, diagnostic_label in all_pairs:
                                model.mode = 'extract_feature'
                                embedding = encode_with_deterministic_tta(
                                    model, sample.cuda(), int(getattr(args, 'encode_tta_views', 1)))
                                joint_layer = str(getattr(args, 'joint_cluster_layer', 'layer4'))
                                if joint_layer in ('layer4', 'layer4_lda'):
                                    joint_feature = embedding
                                elif joint_layer == 'layer4_layer3':
                                    stage_map = model.forward_to_stage(sample.cuda(), stage='layer3', augment=False)
                                    stage_feature = F.adaptive_avg_pool2d(stage_map, 1).flatten(1)
                                    joint_feature = torch.cat([
                                        F.normalize(embedding, dim=1),
                                        float(getattr(args, 'joint_structure_weight', 0.2)) *
                                        F.normalize(stage_feature, dim=1)], dim=1)
                                else:
                                    stage_map = model.forward_to_stage(sample.cuda(), stage=joint_layer, augment=False)
                                    joint_feature = F.adaptive_avg_pool2d(stage_map, 1).flatten(1)
                                seen_count = min(args.num_labeled_classes +
                                                 max(int(session) - 1, 0) * args.num_unlabeled_classes,
                                                 model.fc.weight.shape[0])
                                logits = model.fc(embedding)[:, :seen_count]
                                encoded_pairs.append((embedding.squeeze(0).detach().cpu(), sample,
                                                      diagnostic_label,
                                                      joint_feature.squeeze(0).detach().cpu()))
                                if rank_score == 'encode_mindist':
                                    distance = torch.cdist(embedding, model.fc.weight[:seen_count])
                                    knownness = float((-distance.min(dim=1).values).item())
                                elif rank_score == 'encode_energy':
                                    knownness = float(torch.logsumexp(logits, dim=1).item())
                                else:
                                    knownness = float(logits.max(dim=1).values.item())
                                scored_pairs.append((knownness, sample, diagnostic_label))
                        if rank_score in ('encode_joint_cluster', 'encode_joint_stats', 'encode_joint_cosine'):
                            global _PRECOMPUTED_DISCOVERY_ASSIGNMENTS
                            semantic_x = torch.stack([item[0] for item in encoded_pairs]).numpy()
                            joint_x = torch.stack([item[3] for item in encoded_pairs]).numpy()
                            if str(getattr(args, 'joint_cluster_layer', 'layer4')) == 'layer4_lda':
                                geometry = torch.load(str(args.base_geometry_path), map_location='cpu', weights_only=True)
                                projection = geometry['lda_projection'].cpu().numpy()
                                joint_x = joint_x @ projection
                            if rank_score == 'encode_joint_cosine':
                                joint_x = joint_x / np.clip(np.linalg.norm(joint_x, axis=1, keepdims=True), 1e-12, None)
                            joint_k = args.num_unlabeled_classes * 2
                            capacity = len(joint_x) // joint_k
                            slots = np.repeat(np.arange(joint_k), capacity)
                            best_joint_score = -float('inf')
                            assignments = None
                            trials = max(1, int(getattr(args, 'joint_kmeans_trials', 1)))
                            for trial in range(trials):
                                if str(getattr(args, 'joint_kmeans_random_state', 'seeded')) == 'legacy_none':
                                    joint_random_state = None
                                else:
                                    joint_random_state = (int(getattr(args, 'seed', 0)) +
                                                          int(session or 0) * 1000 + trial)
                                joint = KMeans(
                                    n_clusters=joint_k, n_init=20 if trials == 1 else 1,
                                    random_state=joint_random_state).fit(joint_x)
                                costs = ((joint_x[:, None, :] - joint.cluster_centers_[None, :, :]) ** 2).sum(-1)
                                rows, cols = linear_sum_assignment(costs[:, slots])
                                trial_assignments = np.empty(len(joint_x), dtype=np.int64)
                                trial_assignments[rows] = slots[cols]
                                trial_score = silhouette_score(joint_x, trial_assignments, metric='euclidean')
                                if trial_score > best_joint_score:
                                    best_joint_score = float(trial_score)
                                    assignments = trial_assignments.copy()
                            print(f'  [DISCOVERY-STABILITY] trials={trials} silhouette={best_joint_score:.4f}')
                            centers = np.stack([joint_x[assignments == ci].mean(0)
                                                for ci in range(joint_k)])
                            semantic_centers = np.stack([semantic_x[assignments == ci].mean(0)
                                                        for ci in range(joint_k)])
                            with torch.no_grad():
                                center_tensor = torch.from_numpy(semantic_centers).cuda()
                                if rank_score == 'encode_joint_cosine':
                                    geometry_path = str(getattr(args, 'base_geometry_path', ''))
                                    if not geometry_path or not os.path.isfile(geometry_path):
                                        raise FileNotFoundError('encode_joint_cosine requires local --base_geometry_path')
                                    geometry = torch.load(geometry_path, map_location=device, weights_only=True)
                                    base_centers = F.normalize(geometry['centers'].to(device), dim=1)
                                    cosine_scores = F.normalize(center_tensor, dim=1) @ base_centers.t()
                                    top2 = cosine_scores.topk(2, dim=1).values
                                    margin_weight = float(getattr(args, 'joint_margin_weight', 0.0))
                                    cluster_knownness = (top2[:, 0] + margin_weight *
                                                         (top2[:, 0] - top2[:, 1])).cpu().numpy()
                                elif rank_score == 'encode_joint_stats':
                                    geometry_path = str(getattr(args, 'base_geometry_path', ''))
                                    if not geometry_path or not os.path.isfile(geometry_path):
                                        raise FileNotFoundError('encode_joint_stats requires local --base_geometry_path')
                                    geometry = torch.load(geometry_path, map_location=device, weights_only=True)
                                    base_centers = geometry['centers'].to(device)
                                    radius_mean = geometry['radius_mean'].to(device)
                                    radius_std = geometry['radius_std'].to(device)
                                    distances = torch.cdist(center_tensor, base_centers)
                                    z_distance = (distances - radius_mean.unsqueeze(0)) / radius_std.unsqueeze(0)
                                    cluster_knownness = (-z_distance.min(1).values).cpu().numpy()
                                else:
                                    center_logits = model.fc(center_tensor)[:, :args.num_labeled_classes]
                                    cluster_knownness = center_logits.max(1).values.cpu().numpy()
                            novel_clusters = set(np.argsort(cluster_knownness)[:args.num_unlabeled_classes].tolist())
                            for ci in range(joint_k):
                                diagnostic = [encoded_pairs[idx][2] for idx in range(len(encoded_pairs))
                                              if int(assignments[idx]) == ci]
                                values, counts = np.unique(diagnostic, return_counts=True)
                                top = sorted(zip(values, counts), key=lambda item: -item[1])[:3]
                                print(f'  [JOINT-DIAG] cluster={ci} knownness={cluster_knownness[ci]:.4f} '
                                      f'top_labels={top}')
                            selected = [encoded_pairs[idx] for idx in range(len(encoded_pairs))
                                        if int(assignments[idx]) in novel_clusters]
                            selected_cluster_ids = [int(assignments[idx]) for idx in range(len(encoded_pairs))
                                                    if int(assignments[idx]) in novel_clusters]
                            cluster_remap = {old: new for new, old in enumerate(sorted(novel_clusters))}
                            _PRECOMPUTED_DISCOVERY_ASSIGNMENTS = [cluster_remap[c]
                                                                  for c in selected_cluster_ids]
                            dump_dir = str(getattr(args, 'discovery_dump_dir', ''))
                            if dump_dir:
                                os.makedirs(dump_dir, exist_ok=True)
                                torch.save({
                                    'semantic_features': torch.from_numpy(semantic_x),
                                    'joint_features': torch.from_numpy(joint_x),
                                    'diagnostic_labels': torch.tensor([item[2] for item in encoded_pairs]),
                                    'joint_assignments': torch.from_numpy(assignments),
                                    'selected_indices': torch.tensor([
                                        idx for idx in range(len(encoded_pairs))
                                        if int(assignments[idx]) in novel_clusters]),
                                    'session': int(session), 'labels_used_by_algorithm': False,
                                }, os.path.join(dump_dir, f'session_{int(session)}.pth'))
                            ranked_data = [item[1] for item in selected]
                            ranked_labels = [item[2] for item in selected]
                            print(f'  [DISCOVERY-JOINT] clusters={joint_k} capacity={capacity} '
                                  f'novel_clusters={sorted(novel_clusters)}')
                        else:
                            scored_pairs.sort(key=lambda item: item[0])
                            ranked_data = [item[1] for item in scored_pairs]
                            ranked_labels = [item[2] for item in scored_pairs]
                    else:
                        ranked_data = list(getattr(run_test_fsl, '_last_ranked_novel_data', []))
                        ranked_labels = list(getattr(run_test_fsl, '_last_ranked_novel_labels', []))
                    if ranked_topk > 0 and len(ranked_data) >= ranked_topk:
                        discovery_data = ranked_data[:ranked_topk]
                        discovery_label = ranked_labels[:ranked_topk]
                        print(f'  [DISCOVERY-RANK] selected={ranked_topk}/{len(ranked_data)} '
                              f'by {rank_score} knownness (label-free)')
                    else:
                        discovery_data = unknow_data + know_data
                        discovery_label = unknow_label + know_label
                    print(f'  [DISCOVERY-BUFFER] rejected={len(unknow_data)} '
                          f'accepted_candidates={len(know_data)} total={len(discovery_data)}')
                else:
                    discovery_data, discovery_label = unknow_data, unknow_label
                    reflow_q = float(getattr(args, 'discovery_reflow_quantile', 0.0))
                    boundary = list(getattr(run_test_fsl, '_last_accepted_boundary_distance', []))
                    reflow_active = (int(getattr(args, 'discovery_reflow_start', 1)) <= session <=
                                     int(getattr(args, 'discovery_reflow_end', 999)))
                    rejected_fraction = len(unknow_data) / max(len(unknow_data) + len(know_data), 1)
                    reflow_active = (reflow_active and
                                     float(getattr(args, 'discovery_reflow_reject_min', 0.0)) <= rejected_fraction <=
                                     float(getattr(args, 'discovery_reflow_reject_max', 1.0)))
                    if reflow_q > 0.0 and reflow_active and boundary and len(boundary) == len(know_data):
                        count = min(int(getattr(args, 'discovery_reflow_max', 10)),
                                    max(1, int(np.ceil(len(boundary) * min(reflow_q, 1.0)))))
                        selected = np.argsort(np.asarray(boundary))[:count]
                        discovery_data = discovery_data + [know_data[int(idx)] for idx in selected]
                        discovery_label = discovery_label + [know_label[int(idx)] for idx in selected]
                        print(f'  [DISCOVERY-REFLOW] selected={count}/{len(know_data)} '
                              f'quantile={reflow_q:.3f} rejected={len(unknow_data)} '
                              f'reject_fraction={rejected_fraction:.3f} total={len(discovery_data)}')
                cluster_acc=debug_cluster(args,model,discovery_data,discovery_label,session)
                if getattr(args, 'old_proto_adapt', False):
                    adapt_old_prototypes_after_expansion(args, model, know_data, know_label, session=session)
                if getattr(args, 'joint_proto_refine', False):
                    refine_session_prototypes(args, model, know_data + unknow_data, know_label + unknow_label, session=session)
                acc_known, _ = known_test(args, model, know_data, know_label)
                fscore=calc(args,know_label,unknow_label)
                result['sess{}_ak'.format(session)]+=[acc_known]
                result['sess{}_au'.format(session)]+=[cluster_acc]
                result['sess{}_ar'.format(session)]+=[session_auroc]
                result['sess{}_fs'.format(session)]+=[fscore]
                #incremental learning
                _,testloader = get_testloader(args,session)
                all_acc=test(args, model, testloader,  session)
                _,inc_testloader = get_inc_testloader(args,session)
                inc_acc = test(args, model, inc_testloader,  session)
                result['sess{}_inc'.format(session)]+=[inc_acc]
                result['sess{}_all'.format(session)]+=[all_acc]
                args.num_labeled_classes += args.way
                avg_acc_known = sum(result['sess{}_ak'.format(session)]) / len(result['sess{}_ak'.format(session)])  
                avg_acc_unknown = sum(result['sess{}_au'.format(session)]) / len(result['sess{}_au'.format(session)])  
                avg_auroc = sum(result['sess{}_ar'.format(session)]) / len(result['sess{}_ar'.format(session)])  
                avg_fscore = sum(result['sess{}_fs'.format(session)]) / len(result['sess{}_fs'.format(session)])  
                avg_inc_acc = sum(result['sess{}_inc'.format(session)]) / len(result['sess{}_inc'.format(session)])  
                avg_all_acc = sum(result['sess{}_all'.format(session)]) / len(result['sess{}_all'.format(session)])  
                # Store the raw metric for this repeat. The legacy code stored
                # cumulative means here and then averaged them again, biasing both
                # the reported mean and standard deviation toward early repeats.
                session_ka[i].append(acc_known)
                session_uka[i].append(cluster_acc)
                session_auroc_vals[i].append(session_auroc)
                session_f1s[i].append(fscore)
                session_inc[i].append(inc_acc)
                session_all[i].append(all_acc)
                # avg_session0_acc = sum(session0_acc_list) / len(session0_acc_list)
                # print(f"\n=== Final Average Session 0 Acc: {avg_session0_acc:.4f} ===")
                # result_file.write(f"\nAverage Session 0 Acc: {avg_session0_acc:.4f}\n")
                # 写入文件  
                result_line = "[RAW] round={} session={} acc known={:.6f} acc unknown={:.6f} auroc={:.6f} f1={:.6f} inc={:.6f} all={:.6f}\n".format(
                    i, session, acc_known, cluster_acc, session_auroc, fscore, inc_acc, all_acc)
                result_file.write(result_line)  
                print(result_line.strip())
            best_model_dir = os.path.join(args.save_dir, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=model.state_dict()), best_model_dir)
        session0_acc_values = np.array(session0_acc_list)
        session0_mean = np.mean(session0_acc_values)
        session0_std = np.std(session0_acc_values, ddof=1) if len(session0_acc_values) > 1 else 0.0
        
        print(f"\n=== Final Session 0 ===")
        print(f"Average Acc: {session0_mean:.4f} ± {session0_std:.4f}")
        result_file.write(f"\n=== Final Session 0 ===\n")
        result_file.write(f"Average Acc: {session0_mean:.4f} ± {session0_std:.4f}\n")
        session_ka_means = []  # 存储每个session的 known_acc 均值
        session_uka_means = [] # 存储每个session的 unknown_acc 均值
        session_ar_means = []  # 存储每个session的 auroc 均值
        session_f1s_means = [] # 存储每个session的 f1_score 均值
        session_inc_means = [] # 存储每个session的 incremental_acc 均值
        session_all_means = [] # 存储每个session的 incremental_acc 均值
        for ses in range(args.num_session-1):
            # 计算均值和标准差
            ka_values = [session_ka[time][ses] for time in range(args.test_times)]
            uka_values = [session_uka[time][ses] for time in range(args.test_times)]
            ar_values = [session_auroc_vals[time][ses] for time in range(args.test_times)]
            f1s_values = [session_f1s[time][ses] for time in range(args.test_times)]
            inc_values = [session_inc[time][ses] for time in range(args.test_times)]
            all_values = [session_all[time][ses] for time in range(args.test_times)]
            ka_mean = np.mean(ka_values)
            ka_std = np.std(ka_values, ddof=1) if len(ka_values) > 1 else 0.0
            uka_mean = np.mean(uka_values)
            uka_std = np.std(uka_values, ddof=1) if len(uka_values) > 1 else 0.0
            ar_mean = np.mean(ar_values)
            ar_std = np.std(ar_values, ddof=1) if len(ar_values) > 1 else 0.0
            f1s_mean = np.mean(f1s_values)
            f1s_std = np.std(f1s_values, ddof=1) if len(f1s_values) > 1 else 0.0
            inc_mean = np.mean(inc_values)
            inc_std = np.std(inc_values, ddof=1) if len(inc_values) > 1 else 0.0
            all_mean = np.mean(all_values)
            all_std = np.std(all_values, ddof=1) if len(all_values) > 1 else 0.0
            session_ka_means.append(ka_mean)
            session_uka_means.append(uka_mean)
            session_ar_means.append(ar_mean)
            session_f1s_means.append(f1s_mean)
            session_inc_means.append(inc_mean)
            session_all_means.append(all_mean)
            # 打印带标准差的结果
            print(f"total session{ses+1} acc known is {ka_mean:.4f} ± {ka_std:.4f}")
            print(f"total session{ses+1} acc unknown is {uka_mean:.4f} ± {uka_std:.4f}")
            print(f"total session{ses+1} auroc is {ar_mean:.4f} ± {ar_std:.4f}")
            print(f"total session{ses+1} f1 score is {f1s_mean:.4f} ± {f1s_std:.4f}")
            print(f"total session{ses+1} incremental acc is {inc_mean:.4f} ± {inc_std:.4f}")
            print(f"total session{ses+1} all acc is {all_mean:.4f} ± {all_std:.4f}")
            # 写入文件
            result_row = (
                f"session: {ses+1}, "
                f"total aac known: {ka_mean:.4f} ± {ka_std:.4f}, "
                f"total acc unknown: {uka_mean:.4f} ± {uka_std:.4f}, "
                f"total auroc: {ar_mean:.4f} ± {ar_std:.4f}, "
                f"total f1 score: {f1s_mean:.4f} ± {f1s_std:.4f}, "
                f"total incremental acc: {inc_mean:.4f} ± {inc_std:.4f}, "
                f"total all acc: {all_mean:.4f} ± {all_std:.4f}\n"
            )
            result_file.write(result_row)  
        aa_known = round(np.mean(session_ka_means), 4)
        aa_unknown = round(np.mean(session_uka_means), 4)
        aa_auroc = round(np.mean(session_ar_means), 4)
        aa_f1 = round(np.mean(session_f1s_means), 4)
        aa_inc = round(np.mean(session_inc_means), 4)
        aa_all = round(np.mean(session_all_means), 4)
        print("\n=== Sessions Average Accuracy (AA) ===")
        print(f"Average Acc Known:    {aa_known:.4f}")
        print(f"Average Acc Unknown:  {aa_unknown:.4f}")
        print(f"Average AUROC:        {aa_auroc:.4f}")
        print(f"Average F1 Score:     {aa_f1:.4f}")
        print(f"Average Incremental Acc: {aa_inc:.4f}")
        print(f"Average all Acc: {aa_all:.4f}")
        result_file.write("\n=== Sessions Average Accuracy (AA) ===\n")
        result_file.write(f"Average Acc Known:    {aa_known:.4f}\n")
        result_file.write(f"Average Acc Unknown:  {aa_unknown:.4f}\n")
        result_file.write(f"Average AUROC:        {aa_auroc:.4f}\n")
        result_file.write(f"Average F1 Score:     {aa_f1:.4f}\n")
        result_file.write(f"Average Incremental Acc: {aa_inc:.4f}\n")
        result_file.write(f"Average all Acc: {aa_all:.4f}\n")
        end_idx = len(session_ka_means) - 1
        pd_known = round(session_ka_means[0] - session_ka_means[end_idx], 4)
        pd_unknown = round(session_uka_means[0] - session_uka_means[end_idx], 4)
        pd_f1 = round(session_f1s_means[0] - session_f1s_means[end_idx], 4)
        pd_inc = round(session_inc_means[0] - session_inc_means[end_idx], 4)
        pd_all = round(session_all_means[0] - session_all_means[end_idx], 4)
        # 计算百分比并保留两位小数
        pd_known_pct = round(pd_known * 100, 2)
        pd_unknown_pct = round(pd_unknown * 100, 2)
        pd_f1_pct = round(pd_f1 * 100, 2)
        pd_inc_pct = round(pd_inc * 100, 2)
        pd_all_pct = round(pd_all * 100, 2)
        # 打印性能下降率（PD）
        print("\n=== Performance Degradation (PD: Session1 - SessionLast) ===")
        print(f"PD Acc Known:    {pd_known:.4f} (↓{pd_known_pct}%)")
        print(f"PD Acc Unknown:  {pd_unknown:.4f} (↓{pd_unknown_pct}%)")
        print(f"PD F1 Score:     {pd_f1:.4f} (↓{pd_f1_pct}%)")
        print(f"PD Incremental Acc: {pd_inc:.4f} (↓{pd_inc_pct}%)")
        print(f"PD all Acc: {pd_all:.4f} (↓{pd_all_pct}%)")
        # 写入文件
        result_file.write("\n=== Performance Degradation (PD: Session1 - SessionLast) ===\n")
        result_file.write(f"PD Acc Known:    {pd_known:.4f} (↓{pd_known_pct}%)\n")
        result_file.write(f"PD Acc Unknown:  {pd_unknown:.4f} (↓{pd_unknown_pct}%)\n")
        result_file.write(f"PD F1 Score:     {pd_f1:.4f} (↓{pd_f1_pct}%)\n")
        result_file.write(f"PD Incremental Acc: {pd_inc:.4f} (↓{pd_inc_pct}%)\n")
        result_file.write(f"PD all Acc: {pd_all:.4f} (↓{pd_all_pct}%)\n")
        result_file.close()
def get_class_difficulty(args, model, full_loader):
    """
    [创新点：基于类级不确定性的难度评估]
    计算每个基类样本的平均不确定性，用于排序。
    """
    print("Evaluating Base Class Difficulty...")
    model.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # 记录每个类的累积不确定性和样本数
    class_unc_sum = {}
    class_counts = {}
    
    # 临时开启 MC Dropout
    # 如果 model.get_uncertainty 内部已经处理了 eval/train 切换，这里保持 eval 即可
    
    with torch.no_grad():
        # 只跑一部分数据估算即可，不需要全量，节省时间
        for i, batch in enumerate(tqdm(full_loader, desc="Difficulty Est", leave=False)):
            if i > 200: break # 抽样 200 个 batch
            data, label = [_.to(device) for _ in batch]
            
            # 计算 Batch 不确定性
            # 调用 model.get_uncertainty (支持 batch 计算)
            # n_aug=2, n_forward=2 快速估算
            if hasattr(model, 'module'):
                uncs = model.module.get_uncertainty(data, n_aug=5, n_forward=5)
            else:
                uncs = model.get_uncertainty(data, n_aug=5, n_forward=5)
            
            if isinstance(uncs, torch.Tensor): uncs = uncs.cpu().numpy()
            label = label.cpu().numpy()
            
            for l, u in zip(label, uncs):
                if l not in class_unc_sum:
                    class_unc_sum[l] = 0.0
                    class_counts[l] = 0
                class_unc_sum[l] += u
                class_counts[l] += 1
    
    # 计算平均值
    class_avg_unc = []
    # 确保涵盖所有基类 (0 ~ num_base-1)
    for i in range(args.num_base):
        if i in class_unc_sum and class_counts[i] > 0:
            class_avg_unc.append(class_unc_sum[i] / class_counts[i])
        else:
            class_avg_unc.append(0.0) # 默认简单
            
    # 返回按难度排序的类别索引 (Uncertainty 小 -> 大)
    sorted_classes = np.argsort(class_avg_unc)
    return sorted_classes
from models.uncertainty import get_base_class_uncertainty
def get_initial_difficulty(args, model, full_loader):
    """
    【零样本难度评估】
    利用 ResNet 预训练权重的特征分布来确定初始课程顺序。
    原理：类内特征越紧凑(方差小) -> 越简单；越发散 -> 越困难。
    """
    print("\n=== [Curriculum Init] Sorting classes by Feature Compactness (Zero-shot) ===")
    
    model.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    class_vectors = {} 
    
    # 1. 提取特征 (不经过 FC 层)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(full_loader, desc="Difficulty Est")):
            data, label = [_.to(device) for _ in batch]
            
            # 使用 base_encode 提取特征 (注意 augment=False)
            if hasattr(model, 'module'):
                feats = model.module.base_encode(data, augment=False) 
            else:
                feats = model.base_encode(data, augment=False)
            
            # L2 归一化 (关键：因为后续通常基于 Cosine 相似度)
            feats = F.normalize(feats, p=2, dim=1)
            
            for f, l in zip(feats, label):
                l = l.item()
                if l not in class_vectors: class_vectors[l] = []
                class_vectors[l].append(f.cpu())

    # 2. 计算每个类的“紧密度”
    class_scores = {}
    for cls, feats in class_vectors.items():
        if len(feats) < 2: 
            class_scores[cls] = 0.0
            continue
            
        feats_tensor = torch.stack(feats)
        # 计算类中心
        center = feats_tensor.mean(dim=0, keepdim=True)
        center = F.normalize(center, p=2, dim=1)
        
        # 距离 = 1 - CosineSimilarity (越小越简单)
        distances = 1.0 - torch.mm(feats_tensor, center.t())
        class_scores[cls] = distances.mean().item()

    # 3. 排序：简单 -> 困难
    sorted_classes = np.array(sorted(class_scores, key=class_scores.get))
    
    print(f"[Result] Top-5 Easiest: {sorted_classes[:5]}")
    print(f"[Result] Top-5 Hardest: {sorted_classes[-5:]}")
    
    return sorted_classes
from torch.utils.data import WeightedRandomSampler
from models.uncertainty import get_base_class_uncertainty

# =========================================================================
# 辅助函数：获取数据集的标签列表 (用于构建采样权重)
# =========================================================================
def get_dataset_labels(dataset, num_workers=0):
    # 尝试常见的属性名
    if hasattr(dataset, 'targets'): return np.array(dataset.targets)
    if hasattr(dataset, 'labels'): return np.array(dataset.labels)
    if hasattr(dataset, '_labels'): return np.array(dataset._labels)
    
    # 如果都没有，只能遍历一遍 (稍微花点时间，但为了加权是值得的)
    print("Extracting labels for weighted sampling...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False,
                                         num_workers=num_workers)
    all_labels = []
    for _, y in loader:
        all_labels.append(y)
    return torch.cat(all_labels).cpu().numpy()

# =========================================================================
# 核心函数：base_train (极速课程 + 困难加权版)
# =========================================================================
def base_train(args, model):
    # 1. 准备数据
    full_dataset, full_loader = get_pretrain_dataloader(args) 
    dataset_tag = getattr(args, 'dataset', 'default')
    save_model_path = os.path.join(args.save_dir, f'base_train_for_meta_{dataset_tag}.pth')
    init_checkpoint = str(getattr(args, 'base_init_checkpoint', ''))
    if init_checkpoint:
        if not os.path.isfile(init_checkpoint):
            raise FileNotFoundError(f'base_init_checkpoint is not local: {init_checkpoint}')
        init_state = torch.load(init_checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(init_state.get('params', init_state), strict=True)
        print(f'[BASE-INIT] loaded current encoder from {init_checkpoint}')
    optimizer, scheduler = get_optimizer(model, args)
    finetune_lr = float(getattr(args, 'base_finetune_lr', 0.0))
    if finetune_lr > 0:
        for group in optimizer.param_groups:
            group['lr'] = finetune_lr
    center_loss_fn = UncertaintyCenterLoss(args.num_base, args.feat_dim).cuda()
    optimizer_cent = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5)
    total_epochs = args.epochs.epochs_std
    if bool(getattr(args, 'base_supcon_balanced', False)):
        balanced_sampler = CategoriesSampler(
            full_dataset.targets, n_batch=len(full_loader), n_cls=16, n_per=8)
        full_loader = torch.utils.data.DataLoader(
            full_dataset, batch_sampler=balanced_sampler,
            num_workers=args.dataloader.num_workers, pin_memory=True,
            worker_init_fn=seed_worker)
        print(f'[BASE-SUPCON] class-balanced batches={len(full_loader)} shape=16x8')

    # Clean full-class control.  This is intentionally a separate experimental path:
    # it tests whether curriculum selection/reweighting, rather than the encoder or
    # dataset split, causes a base-session regression.
    if getattr(args, 'base_standard_only', False):
        print(f"\n=== Standard Full-Class Base Training ({total_epochs} Epochs) ===")
        class_index = np.arange(args.num_base)
        if args.dataset == 'FMC':
            val_dataset = args.Dataset.FSDCLIPS(
                root=args.dataroot, phase='val', index=class_index,
                base_sess=True, args=args)
        elif 'nsynth' in args.dataset:
            val_dataset = args.Dataset.NDS(
                root=args.dataroot, phase='val', index=class_index,
                base_sess=True, args=args)
        elif 'librispeech' in args.dataset:
            val_dataset = args.Dataset.LBRS(
                root=args.dataroot, phase='val', index=class_index,
                base_sess=True, args=args)
        else:
            val_dataset = args.Dataset.S2S(
                dataset=args.dataset, root=args.dataroot, phase='val',
                index=class_index, base_sess=True, args=args)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.dataloader.test_batch_size,
            shuffle=False, num_workers=args.dataloader.num_workers,
            pin_memory=True, worker_init_fn=seed_worker, generator=g)
        initial_val = evaluate_base_encoder(args, model, val_loader)
        best_val = initial_val['acc']
        torch.save(dict(params=model.state_dict(), epoch=-1,
                        val_acc=best_val), save_model_path)
        print(f"[BASE-VAL] initial val_acc={best_val:.4f}")
        stale_epochs = 0
        patience = int(getattr(args, 'base_val_patience', 6))
        min_delta = float(getattr(args, 'base_val_min_delta', 1e-4))
        for epoch in range(total_epochs):
            model, metrics = standard_base_train_with_metrics(
                args, model, full_loader, optimizer, scheduler, epoch
            )
            val_metrics = evaluate_base_encoder(args, model, val_loader)
            print(f"[BASE-STANDARD] epoch={epoch} train_acc={metrics['acc']:.4f} "
                  f"train_loss={metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                  f"val_loss={val_metrics['loss']:.4f}")
            if val_metrics['acc'] > best_val + min_delta:
                best_val = val_metrics['acc']
                stale_epochs = 0
                torch.save(dict(params=model.state_dict(), epoch=epoch,
                                val_acc=best_val), save_model_path)
                print(f"[BASE-VAL] new best epoch={epoch} val_acc={best_val:.4f}")
            else:
                stale_epochs += 1
            if scheduler is not None:
                scheduler.step()
            if patience > 0 and stale_epochs >= patience:
                print(f"[BASE-VAL] early stop at epoch={epoch}; best_val={best_val:.4f}")
                break
        checkpoint = torch.load(save_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['params'], strict=True)
        print(f"Base training finished. Best validation model saved to {save_model_path}")
        return save_model_path
    
    
    # -----------------------------------------------------
    # Step 1: 零样本初始排序 (Zero-shot Sort)
    # -----------------------------------------------------
    # 依然保留这个，作为第一阶段的引导
    sorted_classes = get_initial_difficulty(args, model, full_loader)
    
    # -----------------------------------------------------
    # Step 2: 极速课程阶段 (Fast Curriculum) - 前 curriculum_ratio 的轮次
    # -----------------------------------------------------
    # 目标：快速让模型学会简单样本，建立特征骨架，不要浪费太多时间
    curriculum_ratio = getattr(args, 'curriculum_ratio', 0.3)
    curriculum_epochs = int(total_epochs * curriculum_ratio)
    # 分两步走：先练 Top 50% 简单，再练 Top 75%
    phases = [
        (0.50, int(curriculum_epochs * 0.5)), 
        (0.75, int(curriculum_epochs * 0.5))
    ]
    
    current_epoch = 0
    print(f"\n=== [Phase 1] Fast Curriculum Guidance ({curriculum_epochs} Epochs) ===")
    
    for ratio, n_epochs in phases:
        num_keep = int(args.num_base * ratio)
        active_classes = sorted_classes[:num_keep]
        active_classes_idx = np.sort(active_classes)
        
        print(f"--> Training Top {int(ratio*100)}% Easiest Classes ({len(active_classes)})")
        
        # 构造简单类的 Loader
        curr_loader = get_subset_dataloader(args, active_classes_idx)
        
        for _ in range(n_epochs):
            model, _ = standard_base_train_with_metrics(
                args, model, curr_loader, optimizer, scheduler, current_epoch
            )
            current_epoch += 1
            if scheduler is not None: scheduler.step()

    # -----------------------------------------------------
    # Step 3: 困难感知全量训练 (Uncertainty-Weighted Full Training)
    # -----------------------------------------------------
    # 剩下的 70% 轮次，我们训练所有类，但是！给困难类更高的权重！
    print(f"\n=== [Phase 2] Uncertainty-Weighted Full Training ({total_epochs - current_epoch} Epochs) ===")
    
    # A. 重新计算不确定度 (这时候模型已经不是小白了，UNCG 结果很准)
    print(">>> Re-evaluating Difficulty with UNCG...")
    unc_scores = get_base_class_uncertainty(
        model, full_loader, 
        device=torch.device("cuda" if args.cuda else "cpu"), 
        k_dropout=5, a_mask=4
    )
    
    # B. 制作采样权重 (Hard Class Reweighting)
    # 策略：不确定度越高 -> 权重越大
    # 线性映射到 [1, 1 + hard_weight_scale] ，默认 2x，避免 2.5x 过采样稀有类导致梯度抖动
    unc_values = np.array([unc_scores.get(c, 0) for c in range(args.num_base)])
    min_u, max_u = unc_values.min(), unc_values.max()
    scale = getattr(args, 'hard_weight_scale', 1.0)
    class_weights = 1.0 + (unc_values - min_u) / (max_u - min_u + 1e-6) * scale
    class_weights_tensor = torch.tensor(class_weights).float().cuda()
    print(f"Class Weights Map (Top 5 Hardest): {np.sort(class_weights)[-5:]}")
    
    # C. 为每个样本分配权重
    all_labels = get_dataset_labels(full_dataset, args.dataloader.num_workers)
    # 确保 label 都在范围内
    all_labels = all_labels[all_labels < args.num_base] 
    
    samples_weights = torch.tensor([class_weights[l] for l in all_labels], dtype=torch.float)
    
    # D. 创建加权采样器
    weighted_sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    
    # E. 构造带 Sampler 的 Full Loader
    # 注意：使用了 sampler 就不能用 shuffle=True
    weighted_loader = torch.utils.data.DataLoader(
        full_dataset, 
        batch_size=args.dataloader.train_batch_size, 
        sampler=weighted_sampler, # <--- 关键！
        num_workers=args.dataloader.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # F. 全量冲刺训练
    remaining_epochs = total_epochs - current_epoch
    for _ in range(remaining_epochs):
        model, _ = standard_base_train_with_metrics(
            args, model, weighted_loader, optimizer, scheduler, current_epoch,
            center_loss_fn=center_loss_fn,        # 传入 Loss 函数
            optimizer_cent=optimizer_cent,        # 传入 优化器
            class_unc_weights=class_weights_tensor # 传入 权重
        )
        current_epoch += 1
        if scheduler is not None: scheduler.step()

    torch.save(dict(params=model.state_dict()), save_model_path)
    print(f"Base training finished. Model saved to {save_model_path}")
    return save_model_path

# =========================================================
# 别忘了保留下面的辅助函数 (如果你还没定义的话)
# =========================================================
def get_subset_dataloader(args, active_classes_idx):
    if args.dataset == 'FMC':
        curr_dataset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
    elif 'nsynth' in args.dataset:
        curr_dataset = args.Dataset.NDS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
    elif 'librispeech' in args.dataset:
        curr_dataset = args.Dataset.LBRS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
    else:
        curr_dataset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
    
    loader = torch.utils.data.DataLoader(
        curr_dataset, 
        batch_size=args.dataloader.train_batch_size, 
        shuffle=True, 
        num_workers=args.dataloader.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    return loader

@torch.no_grad()
def evaluate_base_encoder(args, model, loader):
    """Evaluate the same model.encode + fc path without touching test data."""
    tl = Averager()
    ta = Averager()
    model.eval()
    for data, labels in loader:
        data, labels = data.cuda(), labels.cuda()
        model.mode = 'extract_feature'
        logits = model.fc(model.encode(data))[:, :args.num_base]
        tl.add(F.cross_entropy(logits, labels).item())
        ta.add(count_acc(logits, labels))
    return {'acc': ta.item(), 'loss': tl.item()}


def standard_base_train_with_metrics(args, model, trainloader, optimizer, scheduler, epoch, 
                                     center_loss_fn=None, optimizer_cent=None, class_unc_weights=None):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # center loss 权重：从 args.center_loss_weight 读取（默认 0.005）
    cw = float(getattr(args, 'center_loss_weight', 0.005))
    
    # [修改 1] 只有当不需要提取纯特征时，才全局设为 encoder
    # 但为了安全起见，我们在 loop 里面手动控制
    # model.mode = 'encoder' 
    
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        # [修改 2] 临时切换模式以获取 512 维纯特征
        # 只要 mode 不是 'encoder'，network.py 里的 encode 就不会过 FC 层
        model.mode = 'extract_feature' 
        features = model.encode(data) # 此时返回 [Batch, 512]
        
        # [修改 3] 手动过 FC 层得到 logits
        logits = model.fc(features)   # 此时得到 [Batch, 100]

        # 1. 基础分类 Loss
        loss_cls = F.cross_entropy(logits, train_label)
        
        # 2. 计算 Center Loss (现在维度匹配了: 512 vs 512)
        loss_center = torch.tensor(0.0).cuda()
        if center_loss_fn is not None and class_unc_weights is not None:
            loss_center = center_loss_fn(features, train_label, class_unc_weights)
        
        acc = count_acc(logits, train_label)
        
        supcon_weight = float(getattr(args, 'base_supcon_weight', 0.0))
        loss_supcon = supervised_contrastive_loss(
            features, train_label,
            temperature=float(getattr(args, 'base_supcon_temperature', 0.1))) if supcon_weight > 0 else features.new_tensor(0.0)
        # 总 Loss
        total_loss = loss_cls + cw * loss_center + supcon_weight * loss_supcon
        
        if scheduler is not None:
            lrc = scheduler.get_last_lr()[0]
        else:
            lrc = optimizer.param_groups[0]['lr']

        tqdm_gen.set_description(
            'Epoch {}, lr={:.4f}, Total={:.4f} (Cls={:.4f}, Cen={:.4f}), Acc={:.4f}'.format(
                epoch, lrc, total_loss.item(), loss_cls.item(), loss_center.item(), acc)
        )
        
        tl.add(total_loss.item())
        ta.add(acc)
        
        optimizer.zero_grad()
        if optimizer_cent: optimizer_cent.zero_grad()
        
        total_loss.backward()
        
        optimizer.step()
        if optimizer_cent:
            # scale-back 梯度以匹配 center loss 的缩放系数
            if cw > 0:
                for param in center_loss_fn.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / cw)
            optimizer_cent.step()
        
    return model, {'acc': ta.item(), 'loss': tl.item()}


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Single-view SupCon over repeated class examples in a standard batch."""
    z = F.normalize(features, dim=1)
    logits = z @ z.t() / max(float(temperature), 1e-6)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return z.new_tensor(0.0)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = (log_prob * positive).sum(1) / positive.sum(1).clamp_min(1)
    return -mean_positive[valid].mean()
# def base_train(args,model):
#     data_dict = {}
#     data_dict['train_set'],data_dict['trainloader']= get_pretrain_dataloader(args) 
#     net_dict = {}
    
#     net_dict['optimizer'], net_dict['scheduler'] = get_optimizer(model, args)
#     save_model_path = os.path.join(args.save_dir, f'base_train_for_meta.pth')
#     #encoder pretrain
    
#     for epoch in range(args.epochs.epochs_std):
#         model=standard_base_train(args, model,data_dict['trainloader'],net_dict['optimizer'], net_dict['scheduler'], epoch) #要不要打印，改对了吗
#         net_dict['epoch'] = epoch
#         net_dict['scheduler'].step()
#     torch.save(dict(params=model.state_dict()), save_model_path)
   
#     return save_model_path

# 别忘了在文件开头导入你写好的 uncertainty 模块

# def standard_base_train(args, model, trainloader, optimizer, scheduler, epoch):
#     num_base = args.num_base
#     tl = Averager()
#     ta = Averager()
#     model = model.train()
#     model.mode = 'encoder'
#     # standard classification for pretrain
#     tqdm_gen = tqdm(trainloader)
#     for i, batch in enumerate(tqdm_gen, 1):
#         data, train_label = [_.cuda() for _ in batch]

#         logits = model(data)
#         loss = F.cross_entropy(logits, train_label)
#         # feat, proj_feat = model.encode(data, return_proj=True)
#         # contrast_loss = model.compute_contrastive_loss(proj_feat[:len(proj_feat)//2], 
#         #                                              proj_feat[len(proj_feat)//2:])
#         # loss = F.cross_entropy(model.fc(feat), train_label)
#         acc = count_acc(logits, train_label)
#         total_loss = loss
#         # total_loss = loss

#         lrc = scheduler.get_last_lr()[0]
#         tqdm_gen.set_description(
#                 'Standard train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
#         tl.add(total_loss.item())
#         ta.add(acc)
        
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#     tl = tl.item()
#     ta = ta.item()
#     print('ta:{},tl:{}'.format(ta,tl))
#     return model


#切换数据集训练时，metatrainer132行和network49行要修改

# def standard_base_train(args, model, trainloader, optimizer, scheduler, epoch):
#     num_base = args.num_base
#     tl = Averager()
#     ta = Averager()
#     model = model.train()
#     model.mode = 'encoder'
#     # standard classification for pretrain
#     tqdm_gen = tqdm(trainloader)
#     for i, batch in enumerate(tqdm_gen, 1):
#         data, train_label = [_.cuda() for _ in batch]

#         logits = model(data)
#         loss = F.cross_entropy(logits, train_label)
#         # feat, proj_feat = model.encode(data, return_proj=True)
#         # contrast_loss = model.compute_contrastive_loss(proj_feat[:len(proj_feat)//2], 
#         #                                              proj_feat[len(proj_feat)//2:])
#         # loss = F.cross_entropy(model.fc(feat), train_label)
#         acc = count_acc(logits, train_label)
#         total_loss = loss
#         # total_loss = loss

#         lrc = scheduler.get_last_lr()[0]
#         tqdm_gen.set_description(
#                 'Standard train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
#         tl.add(total_loss.item())
#         ta.add(acc)
        
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#     tl = tl.item()
#     ta = ta.item()
#     print('ta:{},tl:{}'.format(ta,tl))
#     return model

def print_version_info(model, message):
    print(message)
    for name, param in model.named_parameters():
        
            print(f"{name}: version {param._version}")

if __name__ == '__main__':
    # parse training arguments
    parser = argparse.ArgumentParser('cluster', parents=[args_parser()])
    args = parser.parse_args()
    with open(args.config) as f:           #training configuration file
        cfg = yaml.safe_load(f)
    cfg = cfg['train']
    cli_values = vars(args).copy()
    # ``--save_dir`` is an optional override. Do not let its parser default erase
    # the dataset-specific checkpoint directory loaded from YAML.
    if cli_values.get('save_dir') is None:
        cli_values.pop('save_dir')
    cfg.update(cli_values)
    args = dict2namespace(cfg)
    if getattr(args, 'save_dir', ''):
        os.makedirs(args.save_dir, exist_ok=True)
    if getattr(args, 'eval_repeats', 0) > 0:
        args.test_times = int(args.eval_repeats)
    set_seed(args.seed)  
    args.cuda = torch.cuda.is_available()
    check_randomness()
    train(args)
#切换数据集训练时，metatrainer132行和network49行要修改
