import os
    # 环境变量
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import yaml
import torch
from sklearn.preprocessing import RobustScaler
import torch.nn as nn  
from utils.util import cluster_acc,calc
from utils.utils import *
from network import MYNET,get_optimizer,replace_base_fc
from data.dataloader import *
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from tqdm import tqdm
from openmax import *
from models.metatrainer import meta_train
# from models.metaowtrainer import meta_train
from threshold_free import run_test_fsl
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
from enhance_module import LocalFeatureCluster
import math
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
# class FeatureFusion(nn.Module):
#     def __init__(self, feat_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(feat_dim*2, feat_dim//2),  # 压缩维度
#             nn.ReLU(),
#             nn.Linear(feat_dim//2, 1),          # 输出融合权重
#             nn.Sigmoid()                         # 归一化到[0,1]
#         )
    
#     def forward(self, global_feat, local_feat):
#         # 拼接特征 [B,1024]
#         combined = torch.cat([global_feat, local_feat], dim=1)
#         # 预测动态权重 [B,1]
#         alpha = self.mlp(combined)
#         return alpha * global_feat + (1-alpha) * local_feat
# class EnhancedPositionEncoder(nn.Module):
#     def __init__(self, feat_dim):
#         super().__init__()
#         self.feat_dim = feat_dim
        
#         # 时间轴编码
#         self.time_enc = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(3,1), padding=(1,0)),
#             nn.GELU(),
#             nn.Conv2d(16, feat_dim, kernel_size=(3,1), padding=(1,0))
#         )
        
#         # 频率轴编码
#         self.freq_enc = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=(1,3), padding=(0,1)),
#             nn.GELU(),
#             nn.Conv2d(16, feat_dim, kernel_size=(1,3), padding=(0,1))
#         )
        
#         # 动态融合门控（关键修改）
#         self.gate = nn.Sequential(
#             nn.Conv2d(feat_dim*2, feat_dim//2, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(feat_dim//2, feat_dim, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, grid):
#         # 输入验证
#         assert grid.size(1) == 2, f"输入需要包含2个坐标通道，实际得到{grid.size(1)}"
        
#         # 分别编码
#         time_emb = self.time_enc(grid[:,:1])  # [B,512,H,W]
#         freq_emb = self.freq_enc(grid[:,1:])  # [B,512,H,W]
        
#         # 动态融合
#         combined = torch.cat([time_emb, freq_emb], dim=1)  # [B,1024,H,W]
#         gate = self.gate(combined)  # [B,512,H,W]
        
#         return gate * time_emb + (1-gate) * freq_emb
# class LocalFeatureCluster(nn.Module):
#     def __init__(self, feat_dim=512, k_ratio=0.3, temporal_scale=1.0):
#         super().__init__()
#         self.feat_dim = feat_dim
#         self.k_ratio = k_ratio
#         self.temporal_scale = temporal_scale
#         self._current_device = None
#         self.fusion_net = FeatureFusion(feat_dim)
        
#         # 位置编码器（输入通道2对应x,y坐标）
#         self.pos_encoder = EnhancedPositionEncoder(feat_dim)
#         # self.pos_encoder = nn.Sequential(
#         #     nn.Conv2d(2, 16, kernel_size=3, padding=1),
#         #     nn.ReLU(),
#         #     nn.Conv2d(16, feat_dim, kernel_size=3, padding=1)
#         # )
        
#         # 空间权重网络
#         self.spatial_net = nn.Sequential(
#             nn.Linear(feat_dim, feat_dim//2),
#             nn.ReLU(),
#             nn.Linear(feat_dim//2, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, features):
#         B, C, H, W = features.shape
#         device = features.device
#         self._ensure_device_consistency(device)
        
#         # 1. 位置编码
#         grid = self._generate_grid(H, W, features.device)  # [1,2,H,W]
#         pos_emb = self.pos_encoder(grid)  # [1,C,H,W]
        
#         # 2. 特征增强
#         enhanced_feat = features + pos_emb  # [B,C,H,W]
#         flat_feat = enhanced_feat.flatten(2).permute(0,2,1)  # [B,HW,C]
        
#         # 3. 动态聚类
#         k = max(2, int(H*W*self.k_ratio))
#         clustered_feat = []
#         centers = []
        
#         for i in range(B):
#             # 3.1 构建时序相似度矩阵
#             positions = torch.stack(torch.meshgrid(
#                 torch.arange(H, device=device),
#                 torch.arange(W, device=device),
#                 indexing='ij'
#             ), dim=-1).float().view(-1,2)  # [HW,2]
#             temporal_sim = torch.exp(-torch.cdist(positions, positions)/self.temporal_scale)
            
#             with torch.no_grad():
#                 # 3.2 执行聚类
#                 kmeans = KMeans(n_clusters=k, n_init=10).fit(flat_feat[i].cpu().numpy())
#                 center = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
#                 label = torch.from_numpy(kmeans.labels_).to(device)
            
#             # 3.3 时序加权原型
#             for c in range(k):
#                 mask = (label == c)
#                 if mask.sum() > 0:
#                     weights = temporal_sim[mask][:, mask].mean(1)
#                     center[c] = (flat_feat[i][mask] * weights.unsqueeze(1)).sum(0) / (weights.sum() + 1e-6)
            
#             clustered_feat.append(center[label])
#             centers.append(center)
        
#         # 4. 特征融合
#         clustered_feat = torch.stack(clustered_feat)  # [B,HW,C]
#         centers = torch.stack(centers)  # [B,k,C]
        
#         # 5. 空间权重增强
#         spatial_weight = self.spatial_net(flat_feat)  # [B,HW,1]
#         enhanced = spatial_weight * clustered_feat + (1-spatial_weight) * flat_feat  # [B,HW,C]
#         global_feat = features.mean(dim=[2,3])    # [B,512]
#         local_feat = enhanced.mean(dim=[1])# [B,512]
#         enhanced_feat = self.fusion_net(global_feat, local_feat)
#         return enhanced_feat, centers
#     def _generate_grid(self, H, W, device):
#         y_coords = torch.linspace(-1, 1, H, device=device)
#         x_coords = torch.linspace(-1, 1, W, device=device)
#         grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
#         return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1,2,H,W]
#     def _ensure_device_consistency(self, device):
#         if self._current_device != device:
#             self.to(device)
#             self._current_device = device
#             # 验证关键参数设备
#             assert next(self.pos_encoder.parameters()).device == device
#             assert next(self.spatial_net.parameters()).device == device
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
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    parser.add_argument('--num_labeled_classes', default=80, type=int)
    parser.add_argument('--checkpoint', type=bool, default=False)
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
    # 在args_parser()中添加以下参数
    parser.add_argument('--pit_weight', type=float, default=0.5, help='weight for pseudo-incremental loss')
    parser.add_argument('--pit_num_new_classes', type=int, default=5, help='number of pseudo new classes')
    parser.add_argument('--pit_base_momentum', type=float, default=0.7, help='momentum for base class weight update')
    parser.add_argument('--pit_mixup_alpha', type=float, default=0.5, help='alpha for mixup augmentation')
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
# 文件: baseline/test.py

# def debug_cluster(args, model, data, labels, session=None):
#     """
#     修正版 UNCG 聚类方案：
#     1. 标准 K-Means++ 聚类 (保证簇的多样性和几何合理性)
#     2. L2 特征归一化 (适配余弦相似度度量)
#     3. 不确定性加权更新 (抑制噪声样本对原型的污染)
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 1. 初始化工具
#     # 确保 enhancer 和 model 在同一设备
#     local_cluster = LocalFeatureCluster(k_ratio=0.4).to(device)
    
#     # 2. 计算每个样本的不确定性 (Uncertainty Estimation)
#     uncertainties = []
#     # print(f"Session {session}: Calculating uncertainty for {len(data)} samples...")
    
#     # 这里的 calculate_uncertainty_unlabeled 保持你现在的版本即可
#     # 注意：内部调用了 model.hgnn_encode(..., augment=True)
#     for sample in data:
#         u = calculate_uncertainty_unlabeled(
#             model, local_cluster, sample, 
#             n_aug=5, n_forward=5
#         )
#         uncertainties.append(u)
    
#     uncertainties = np.array(uncertainties)
    
#     # 3. 计算重要性权重
#     # 归一化权重，使后续计算更稳健
#     weights = 1.0 / (uncertainties + 1e-8)
#     # 这里不需要全局归一化，我们在簇内更新时会再归一化
    
#     # 4. 提取用于聚类的标准特征
#     model.eval() 
#     features_list = []
#     with torch.no_grad():
#         for x in data:
#             x = x.unsqueeze(0) if x.dim()==1 else x
#             f = model.hgnn_encode(x.to(device)) # [1, 512, H, W]
#             f, _ = local_cluster(f)             # [1, 512]
#             features_list.append(f.squeeze())
            
#     features = torch.stack(features_list) # [N, 512]
    
#     # 【关键优化 1】特征归一化
#     # 因为你的分类器是 Cosine Classifier，聚类应基于方向而非模长
#     features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
#     features_np = features_norm.cpu().numpy()
    
#     # 【关键优化 2】回归标准 K-Means 初始化
#     # 放弃 top_k_indices 初始化，避免所有中心挤在一个类里
#     kmeans = KMeans(
#         n_clusters=args.num_unlabeled_classes, 
#         init='k-means++',  # 使用标准初始化保证中心分散
#         n_init=20,         # 增加尝试次数以获得更优解
#         random_state=args.seed
#     ).fit(features_np)
    
#     y = kmeans.labels_
    
#     # 计算聚类准确率 (仅用于评估)
#     acc, map_dict = cluster_acc(args, np.array(labels), y)
    
#     # 5. 加权原型更新 (Weighted Prototype Update)
#     updated = 0
#     for cluster_id in np.unique(y):
#         if cluster_id in map_dict:
#             true_label = map_dict[cluster_id]
            
#             # 仅处理新类
#             if true_label >= args.num_labeled_classes:
#                 indices = np.where(y == cluster_id)[0]
                
#                 if len(indices) > 0:
#                     # 获取该簇内样本的 特征(未归一化) 和 权重
#                     # 更新原型时建议用未归一化的原始特征，保留强度信息
#                     cluster_feats = features[indices]      
#                     cluster_weights = torch.tensor(weights[indices]).to(device)
                    
#                     # 【关键优化 3】簇内权重归一化
#                     # 确保 sum(weights) = 1，实现加权平均
#                     cluster_weights = cluster_weights / cluster_weights.sum()
                    
#                     # 加权平均计算新原型
#                     # 高确定性(高权重)样本对原型的贡献更大
#                     new_proto = torch.sum(cluster_feats * cluster_weights.unsqueeze(1), dim=0)
                    
#                     # 更新模型参数
#                     model.fc.weight.data[true_label] = new_proto
#                     updated += 1
    
#     # print(f"Session {session}: acc={acc:.4f}, updated={updated}")
#     return acc
def debug_cluster(args, model, data, labels, session=None):
    """改进的特征聚类函数（带时序约束）"""
    with torch.no_grad():
        features = torch.stack([model.hgnn_encode(x).squeeze() for x in data])  # [N,512,H,W]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features,_ = LocalFeatureCluster(k_ratio=0.4)(features)
        features = features.to(device)
        kmeans = KMeans(n_clusters=args.num_unlabeled_classes, n_init=20).fit(features.cpu().numpy())
        
        # 原型更新
        y = kmeans.labels_
        acc, map = cluster_acc(args, np.array(labels), y)
        
        updated = 0
        for cluster_id in np.unique(y):
            if cluster_id in map:
                true_label = map[cluster_id]
                if true_label >= args.num_labeled_classes:
                    indices = np.where(y == cluster_id)[0]
                    if len(indices) > 0:
                        new_proto =features[indices].mean(dim=0).to('cuda')  # 使用压缩后的特征
                        model.fc.weight.data[true_label] = new_proto
                        updated += 1
    
    return acc
# def debug_cluster(args, model, data, labels, session=None):
#     """使用融合特征作为聚类输入的改进版"""
#     # 1. 特征提取与增强
#     with torch.no_grad():
#         features = torch.stack([model.hgnn_encode(x).squeeze() for x in data])  # [N,512,13,4]
        
#         # 保存原始特征用于可视化
#         original_features = features.mean(dim=[2,3]).cpu().numpy()  # [N,512]
        
#         # 全局和局部特征提取
#         global_feat = features.mean(dim=[2,3])  # [N,512]
#         clustered_feat, _ = LocalFeatureCluster(k_ratio=0.5 + session*0.1)(features)  # [N,512,13,4]
#         local_feat = clustered_feat.mean(dim=[2,3])  # [N,512]
        
#         # 动态加权融合
#         alpha = min(0.7, 0.5 + session*0.1)  # 动态调整权重
#         enhanced_feat = alpha * global_feat + (1-alpha) * local_feat  # [N,512]
#         enhanced_features = enhanced_feat.cpu().numpy()
    
#     # 2. 可视化特征分布
    
#     # 3. 直接使用融合特征进行聚类
#     kmeans = KMeans(
#         n_clusters=args.num_unlabeled_classes,
#         n_init=20,
#         random_state=args.seed
#     ).fit(enhanced_features)
#     y = kmeans.labels_
    
#     # 4. 计算评估指标
#     labels = np.array(labels)
#     acc, map = cluster_acc(args, labels, y)
    
#     # 5. 原型更新（仅更新新类别）
#     updated = 0
#     for cluster_id in np.unique(y):
#         if cluster_id in map:
#             true_label = map[cluster_id]
#             if true_label >= args.num_labeled_classes:
#                 indices = np.where(y == cluster_id)[0]
#                 if len(indices) > 0:
#                     with torch.no_grad():
#                         new_proto = enhanced_feat[indices].mean(dim=0)
#                         model.fc.weight.data[true_label] = new_proto
#                         updated += 1
    
#     print(f"Session {session}: acc={acc:.4f}, updated={updated} (alpha={alpha:.2f})")
#     return acc

# def debug_cluster(args, model, data, labels, session=None):
#     """使用融合特征作为聚类输入的改进版"""
#     # 1. 特征提取与增强
#     with torch.no_grad():
#         features = torch.stack([model.hgnn_encode(x).squeeze() for x in data])  # [N,512,13,4]
        
#         # 全局和局部特征提取
#         global_feat = features.mean(dim=[2,3])  # [N,512]
#         clustered_feat, _ = LocalFeatureCluster(k_ratio=0.5 + session*0.1)(features)  # [N,512,13,4]
#         local_feat = clustered_feat.mean(dim=[2,3])  # [N,512]
        
#         # 动态加权融合
#         alpha = min(0.7, 0.5 + session*0.1)  # 动态调整权重
#         enhanced_feat = alpha * global_feat + (1-alpha) * local_feat  # [N,512]
    
#     # 2. 直接使用融合特征进行聚类
#     features_np = enhanced_feat.cpu().numpy()
#     kmeans = KMeans(
#         n_clusters=args.num_unlabeled_classes,
#         n_init=20,
#         random_state=args.seed
#     ).fit(features_np)
#     y = kmeans.labels_
    
#     # 3. 计算评估指标
#     labels = np.array(labels)
#     acc, map = cluster_acc(args,labels, y)
    
#     # 4. 原型更新（仅更新新类别）
#     updated = 0
#     for cluster_id in np.unique(y):
#         if cluster_id in map:
#             true_label = map[cluster_id]
#             if true_label >= args.num_labeled_classes:
#                 indices = np.where(y == cluster_id)[0]
#                 if len(indices) > 0:
#                     with torch.no_grad():
#                         new_proto = enhanced_feat[indices].mean(dim=0)
#                         model.fc.weight.data[true_label] = new_proto
#                         updated += 1
    
#     # 5. 可视化验证
#     visualize_enhanced_cluster(
#         features_np, y, labels,
#         save_path=f"{args.save_dir}/session_{session}_cluster.png"
#     )
    
#     print(f"Session {session}: acc={acc:.4f}, updated={updated} (alpha={alpha:.2f})")
#     return acc
# def visualize_enhanced_cluster(original_features, enhanced_features, pred_labels, true_labels, save_path):
#     """可视化原始特征和增强特征的分布对比"""
#     plt.figure(figsize=(24, 6))
    
#     # 动态调整perplexity
#     n_samples = len(original_features)
#     perplexity = min(30, n_samples - 1)
    
#     # 1. 原始特征可视化
#     plt.subplot(141)
#     try:
#         orig_vis = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(original_features)
#         plt.scatter(orig_vis[:,0], orig_vis[:,1], c=pred_labels, cmap='tab20', alpha=0.6)
#         plt.title("Original Features (t-SNE)")
#     except:
#         orig_vis = PCA(n_components=2).fit_transform(original_features)
#         plt.scatter(orig_vis[:,0], orig_vis[:,1], c=pred_labels, cmap='tab20', alpha=0.6)
#         plt.title("Original Features (PCA)")
    
#     # 2. 增强特征可视化
#     plt.subplot(142)
#     try:
#         enh_vis = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(enhanced_features)
#         plt.scatter(enh_vis[:,0], enh_vis[:,1], c=pred_labels, cmap='tab20', alpha=0.6)
#         plt.title("Enhanced Features (t-SNE)")
#     except:
#         enh_vis = PCA(n_components=2).fit_transform(enhanced_features)
#         plt.scatter(enh_vis[:,0], enh_vis[:,1], c=pred_labels, cmap='tab20', alpha=0.6)
#         plt.title("Enhanced Features (PCA)")
    
#     # 3. 原始特征分布直方图
#     plt.subplot(143)
#     plt.hist(original_features.flatten(), bins=50, alpha=0.7, color='blue')
#     plt.title("Original Feature Distribution")
#     plt.xlabel("Feature Value")
#     plt.ylabel("Frequency")
    
#     # 4. 增强特征分布直方图
#     plt.subplot(144)
#     plt.hist(enhanced_features.flatten(), bins=50, alpha=0.7, color='orange')
#     plt.title("Enhanced Feature Distribution")
#     plt.xlabel("Feature Value")
#     plt.ylabel("Frequency")
    
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
# def debug_cluster(args, model, data, labels, session=None):
#     """ 
#     改进版层次化聚类方案B++
#     核心改进：
#     1. 混合全局+局部特征增强
#     2. 置信度加权投票
#     3. 弹性原型更新
#     """
#     # 1. 特征提取与增强
#     with torch.no_grad():
#         # 获取原始特征 [N,512,13,4]
#         features = torch.stack([model.hgnn_encode(x).squeeze() for x in data])
        
#         # 全局平均特征 [N,512]
#         global_feat = features.mean(dim=[2,3])  
        
#         # 局部聚类增强（增大k_ratio保留更多细节）
#         if session==4:
#             cluster = LocalFeatureCluster(k_ratio=0.8)  # 原为0.25
#         elif session==2:
#             cluster = LocalFeatureCluster(k_ratio=0.7)  # 原为0.25
#         elif session==3:
#             cluster = LocalFeatureCluster(k_ratio=0.7)  # 原为0.25
#         else:
#             cluster = LocalFeatureCluster(k_ratio=0.5)  # 原为0.25
#         # cluster = LocalFeatureCluster(k_ratio=0.75)  # 原为0.25
#         clustered_feat, centroids = cluster(features)  # [N,512,13,4], [N,k,512]
#         local_feat = clustered_feat.mean(dim=[2,3])  # [N,512]
        
#         # 动态加权融合特征 [N,512]
#         alpha = min(0.7, 0.5 + session*0.1)  # 随session增加局部特征权重
#         enhanced_feat = alpha * global_feat + (1-alpha) * local_feat
        
#     # 2. 全局聚类（固定簇数）
#     all_centroids = centroids.view(-1, centroids.shape[-1]).cpu().numpy()
#     global_kmeans = KMeans(
#         n_clusters=args.num_unlabeled_classes,
#         n_init=20,  # 增加初始化次数
#         random_state=args.seed
#     ).fit(all_centroids)
    
#     # 3. 置信度加权标签分配
#     y = []
#     for i in range(len(data)):
#         # 计算每个局部中心的权重（距离倒数）
#         dists = pairwise_distances(
#             centroids[i].cpu().numpy(),
#             global_kmeans.cluster_centers_,
#             metric='cosine'  # 改用余弦相似度
#         )
#         weights = 1 / (dists + 1e-6)
#         weighted_vote = np.argmax(weights.sum(axis=0))
#         y.append(weighted_vote)
#     y = np.array(y)
    
#     # 4. 弹性原型更新
#     labels = np.array(labels)
#     acc, map = cluster_acc(args, labels, y)
    
#     updated = 0
#     for cluster_id in np.unique(y):
#         if cluster_id in map:
#             true_label = map[cluster_id]
#             if true_label >= args.num_labeled_classes:
#                 indices = np.where(y == cluster_id)[0]
#                 if len(indices) > 0:
#                     with torch.no_grad():
#                         # 计算新原型
#                         new_proto = torch.stack(
#                             [model.encode(data[i]) for i in indices]
#                         ).mean(dim=0)
                        
#                         # 动量更新（缓解灾难性遗忘）
#                         momentum = 0.7 if session > 0 else 0.3  # 增量阶段更高动量
#                         model.fc.weight.data[true_label] = (
#                             momentum * model.fc.weight.data[true_label] +
#                             (1-momentum) * new_proto
#                         )
#                         updated += 1
    
#     print(f"Session {session}: acc_unknown={acc:.4f}, updated={updated} "
#           f"(alpha={alpha:.2f}, momentum={momentum:.2f})")
#     return acc


# baseline
# def debug_cluster(args,model,data,labels,session=None):
#     u_features,true_label,feat=[],[],[]
#     x=0
#     for i in range(len(data)):
#         u_feature = model.encode(data[i]).squeeze()
#         u_features.append(u_feature.detach().cpu().numpy())
#     # print(u_features)
#     kmeans = KMeans(n_clusters=args.num_unlabeled_classes,n_init=20).fit(u_features)         
#     y = kmeans.labels_
#     labels = np.array(labels)
#     acc,map=cluster_acc(args,labels,y)
#     x = len(map)
#     if x>5:
#         x=5
#     for j in range(x):
#         indexs=np.where(y==j)
#         if j in map:
#             true_label =map[j]#labels[indexs]s
#             if true_label>=args.num_labeled_classes:
#                 for ind in range(len(indexs[0])):
#                     feat.append(model.encode(data[indexs[0][ind]]))
#                 feat = torch.cat(feat,dim=0)
#                 value_feat = feat.mean(0)
#                 model.fc.weight.data[true_label,:] = value_feat
#                 feat=[]
#     return acc


def test(args, model, testloader,  session):    
    test_class = args.num_base + session * args.way
    model = model.eval()
    num_batch=0
    va=0.0
    sup_emb, novel_ids = None, None
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            model.mode = 'incre'
            query = model.encode(data)
            # query,_ = LocalFeatureCluster(k_ratio=0.3)(query)
            # print(f"Original query shape: {query.shape}")
            proto = model.fc.weight[:test_class, :].detach()
            logits=F.cosine_similarity(query.unsqueeze(1), proto, dim=-1)
            acc = count_acc(logits, test_label)
            num_batch+=1
            va+=acc
    return float(va/num_batch)

#baseline
def known_test(model,data,label):
    feats=[]
    label = torch.tensor(label)
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(len(data)):
        feat = model.hgnn_encode(data[i])
        feat,_ = LocalFeatureCluster(k_ratio=0.4)(feat)
        feat = feat.to(device)
        feats.append(feat)
    proto = model.fc.weight[:args.num_labeled_classes,:].detach().unsqueeze(0).unsqueeze(0)
    feats = torch.stack(feats)
    logits=F.cosine_similarity(feats, proto, dim=-1)
    logits=torch.squeeze(logits)
    acc = count_acc(logits, label.to('cuda'))
    preds = torch.argmax(logits, dim=1)
    score = f1_score(label.cpu().numpy(),preds.cpu().numpy(),average='macro')
    return acc,score

def train(args: dict):   
    # ============ base session training ==============
    device = torch.device("cuda" if args.cuda else "cpu")
    model = MYNET(args, mode='encoder')
    model = model.to(device)
    model.apply(weights_init)  # 使用固定种子的初始化
    set_up_datasets(args)
    if args.checkpoint:
        best_model_dir = args.save_dir+'/'+'epoch_5.pth'
        #meta-train negative prototype
        params = torch.load(best_model_dir, weights_only=True)['cls_params']
        cls_params = {k: v for k, v in params.items() if 'fc' in k}
        model.cls_classifier.init_representation(cls_params)
        model_dict = model.state_dict()
        model_dict.update(params)
        model.load_state_dict(model_dict)
    else:
        best_model_dir=base_train(args,model)
        # best_model_dir=os.path.join(args.save_dir, f'base_train_for_meta.pth')#
        # state_dict = torch.load(best_model_dir)
        # model.load_state_dict(state_dict['params'], strict=True)
        open_train_val_loader= get_dataloaders(args,'openmeta')
        meta_train(args, model,open_train_val_loader, eval_loader=None)
    data_dict,result={},{}
    data_dict['train_set'],_=get_pretrain_dataloader(args)
    model = replace_base_fc(args,data_dict['train_set'], model) 
    with open(os.path.join(args.save_result,'test_result.txt'),'w')as result_file:
        session0_acc_list = []
        session_ka = [[] for _ in range(args.test_times)]
        session_uka = [[] for _ in range(args.test_times)]
        session_f1s = [[] for _ in range(args.test_times)]
        session_inc = [[] for _ in range(args.test_times)]
        session_all = [[] for _ in range(args.test_times)]
        for j in range(0, args.num_session): 
            result['sess{}_ak'.format(j)]=[]
            result['sess{}_au'.format(j)]=[]
            result['sess{}_fs'.format(j)]=[]
            result['sess{}_inc'.format(j)]=[]
            result['sess{}_all'.format(j)]=[]
        for i in range(args.test_times):
            args.current_test = i  # 记录当前测试轮次
            args.num_labeled_classes = args.num_base
            print(f"\n=== Base Session Pure Evaluation (Round {i}) ===")
            _, base_testloader = get_testloader(args, 0)  
            base_acc = test(args, model, base_testloader, 0)  
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
                _,unlabelled_loader = get_dataloader(args, session) #已知5类+未知5类
                #OSR_DETECTION
                unknow_data,unknow_label,know_data,know_label=run_test_fsl(model,args,unlabelled_loader)
                #K means
                cluster_acc=debug_cluster(args,model,unknow_data,unknow_label,session)
                acc_known,_ = known_test(model,know_data,know_label)
                fscore=calc(args,know_label,unknow_label)
                result['sess{}_ak'.format(session)]+=[acc_known]
                result['sess{}_au'.format(session)]+=[cluster_acc]
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
                avg_fscore = sum(result['sess{}_fs'.format(session)]) / len(result['sess{}_fs'.format(session)])  
                avg_inc_acc = sum(result['sess{}_inc'.format(session)]) / len(result['sess{}_inc'.format(session)])  
                avg_all_acc = sum(result['sess{}_all'.format(session)]) / len(result['sess{}_all'.format(session)])  
                session_ka[i].append(avg_acc_known)
                session_uka[i].append(avg_acc_unknown)
                session_f1s[i].append(avg_fscore)
                session_inc[i].append(avg_inc_acc)
                session_all[i].append(avg_all_acc)
                # avg_session0_acc = sum(session0_acc_list) / len(session0_acc_list)
                # print(f"\n=== Final Average Session 0 Acc: {avg_session0_acc:.4f} ===")
                # result_file.write(f"\nAverage Session 0 Acc: {avg_session0_acc:.4f}\n")
                # 写入文件  
                result_line = "session: {}, aac known: {:.4f}, acc unknown: {:.4f}, f1 score: {:.4f}, inc acc: {:.4f}, all acc: {:.4f}\n".format(  
                    session, avg_acc_known, avg_acc_unknown, avg_fscore, avg_inc_acc,avg_all_acc)  
                result_file.write(result_line)  
                print("session:{},acc known:{:.4f},acc unknown:{:.4f},f1 score:{:.4f},incremental acc:{:.4f},all acc:{:.4f}".format(session,(sum(result['sess{}_ak'.format(session)])/len(result['sess{}_ak'.format(session)])), 
           (sum(result['sess{}_au'.format(session)])/len(result['sess{}_au'.format(session)])),(sum(result['sess{}_fs'.format(session)])/len(result['sess{}_fs'.format(session)])), sum(result['sess{}_inc'.format(session)])/len(result['sess{}_inc'.format(session)]),(sum(result['sess{}_all'.format(session)]) / len(result['sess{}_all'.format(session)]) )))
            best_model_dir = os.path.join(args.save_dir, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=model.state_dict()), best_model_dir)
        session0_acc_values = np.array(session0_acc_list)
        session0_mean = np.mean(session0_acc_values)
        session0_std = np.std(session0_acc_values)
        
        print(f"\n=== Final Session 0 ===")
        print(f"Average Acc: {session0_mean:.4f} ± {session0_std:.4f}")
        result_file.write(f"\n=== Final Session 0 ===\n")
        result_file.write(f"Average Acc: {session0_mean:.4f} ± {session0_std:.4f}\n")
        session_ka_means = []  # 存储每个session的 known_acc 均值
        session_uka_means = [] # 存储每个session的 unknown_acc 均值
        session_f1s_means = [] # 存储每个session的 f1_score 均值
        session_inc_means = [] # 存储每个session的 incremental_acc 均值
        session_all_means = [] # 存储每个session的 incremental_acc 均值
        for ses in range(args.num_session-1):
            # 计算均值和标准差
            ka_values = [session_ka[time][ses] for time in range(args.test_times)]
            uka_values = [session_uka[time][ses] for time in range(args.test_times)]
            f1s_values = [session_f1s[time][ses] for time in range(args.test_times)]
            inc_values = [session_inc[time][ses] for time in range(args.test_times)]
            all_values = [session_all[time][ses] for time in range(args.test_times)]
            ka_mean = np.mean(ka_values)
            ka_std = np.std(ka_values)
            uka_mean = np.mean(uka_values)
            uka_std = np.std(uka_values)
            f1s_mean = np.mean(f1s_values)
            f1s_std = np.std(f1s_values)
            inc_mean = np.mean(inc_values)
            inc_std = np.std(inc_values)
            all_mean = np.mean(all_values)
            all_std = np.std(all_values)
            session_ka_means.append(round(ka_mean,4))
            session_uka_means.append(round(uka_mean,4))
            session_f1s_means.append(round(f1s_mean,4))
            session_inc_means.append(round(inc_mean,4))
            session_all_means.append(round(all_mean,4))
            # 打印带标准差的结果（保持原有打印格式）
            print(f"total session{ses+1} acc known is {ka_mean:.4f} ± {ka_std:.4f}")
            print(f"total session{ses+1} acc unknown is {uka_mean:.4f} ± {uka_std:.4f}")
            print(f"total session{ses+1} f1 score is {f1s_mean:.4f} ± {f1s_std:.4f}")
            print(f"total session{ses+1} incremental acc is {inc_mean:.4f} ± {inc_std:.4f}")
            print(f"total session{ses+1} all acc is {all_mean:.4f} ± {all_std:.4f}")
            # 写入文件（保持原有格式）
            result_row = (
                f"session: {ses+1}, "
                f"total aac known: {ka_mean:.4f} ± {ka_std:.4f}, "
                f"total acc unknown: {uka_mean:.4f} ± {uka_std:.4f}, "
                f"total f1 score: {f1s_mean:.4f} ± {f1s_std:.4f}, "
                f"total incremental acc: {inc_mean:.4f} ± {inc_std:.4f}, "
                f"total all acc: {all_mean:.4f} ± {all_std:.4f}\n"
            )
            result_file.write(result_row)  
        aa_known = round(np.mean(session_ka_means), 4)    # known_acc的平均准确率
        aa_unknown = round(np.mean(session_uka_means), 4) # unknown_acc的平均准确率
        aa_f1 = round(np.mean(session_f1s_means), 4)      # f1_score的平均准确率
        aa_inc = round(np.mean(session_inc_means), 4)     # incremental_acc的平均准确率    
        aa_all = round(np.mean(session_all_means), 4)
        print("\n=== 4 Sessions Average Accuracy (AA) ===")
        print(f"Average Acc Known:    {aa_known:.4f}")
        print(f"Average Acc Unknown:  {aa_unknown:.4f}")
        print(f"Average F1 Score:     {aa_f1:.4f}")
        print(f"Average Incremental Acc: {aa_inc:.4f}")
        print(f"Average all Acc: {aa_all:.4f}")
        result_file.write("\n=== 4 Sessions Average Accuracy (AA) ===\n")
        result_file.write(f"Average Acc Known:    {aa_known:.4f}\n")
        result_file.write(f"Average Acc Unknown:  {aa_unknown:.4f}\n")
        result_file.write(f"Average F1 Score:     {aa_f1:.4f}\n")
        result_file.write(f"Average Incremental Acc: {aa_inc:.4f}\n")
        result_file.write(f"Average all Acc: {aa_all:.4f}\n")
        pd_known = round(session_ka_means[0] - session_ka_means[3], 4)    # known_acc的性能下降
        pd_unknown = round(session_uka_means[0] - session_uka_means[3], 4) # unknown_acc的性能下降
        pd_f1 = round(session_f1s_means[0] - session_f1s_means[3], 4)      # f1_score的性能下降
        pd_inc = round(session_inc_means[0] - session_inc_means[3], 4)     # incremental_acc的性能下降
        pd_all = round(session_all_means[0] - session_all_means[3], 4)     # incremental_acc的性能下降
        # 计算百分比并保留两位小数
        pd_known_pct = round(pd_known * 100, 2)
        pd_unknown_pct = round(pd_unknown * 100, 2)
        pd_f1_pct = round(pd_f1 * 100, 2)
        pd_inc_pct = round(pd_inc * 100, 2)
        pd_all_pct = round(pd_all * 100, 2)
        # 打印性能下降率（PD）
        print("\n=== Performance Degradation (PD: Session1 - Session4) ===")
        print(f"PD Acc Known:    {pd_known:.4f} (↓{pd_known_pct}%)")
        print(f"PD Acc Unknown:  {pd_unknown:.4f} (↓{pd_unknown_pct}%)")
        print(f"PD F1 Score:     {pd_f1:.4f} (↓{pd_f1_pct}%)")
        print(f"PD Incremental Acc: {pd_inc:.4f} (↓{pd_inc_pct}%)")
        print(f"PD all Acc: {pd_all:.4f} (↓{pd_all_pct}%)")
        # 写入文件
        result_file.write("\n=== Performance Degradation (PD: Session1 - Session4) ===\n")
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
from models.uncertainty import get_base_class_uncertainty

def base_train(args, model):
    # 1. 准备全量数据
    full_dataset, full_loader = get_pretrain_dataloader(args) 
    save_model_path = os.path.join(args.save_dir, 'base_train_for_meta.pth')
    optimizer, scheduler = get_optimizer(model, args)
    
    # 定义 Warm-up 轮数
    WARMUP_EPOCHS = 5 
    total_epochs = args.epochs.epochs_std
    
    current_epoch = 0

    # ====================================================
    # Step 1: Warm-up 阶段 (全量随机训练)
    # ====================================================
    print(f"\n=== [Step 1] Warm-up Phase ({WARMUP_EPOCHS} Epochs) ===")
    
    for epoch in range(WARMUP_EPOCHS):
        # 【修正】移除多余参数，匹配 standard_base_train(args, model, trainloader, optimizer, scheduler, epoch)
        model = standard_base_train(
            args, model, full_loader, optimizer, scheduler, current_epoch
        )
        current_epoch += 1
        if scheduler is not None: scheduler.step()

    # ====================================================
    # Step 2: 计算 UNCG 不确定度并排序
    # ====================================================
    print(f"\n=== [Step 2] Calculating UNCG Uncertainty ===")
    
    unc_scores = get_base_class_uncertainty(
        model, full_loader, 
        device=torch.device("cuda" if args.cuda else "cpu"), 
        k_dropout=5, a_mask=4
    )
    
    # 排序：简单 -> 困难
    sorted_classes = np.array(sorted(unc_scores, key=unc_scores.get))
    print(f"[UNCG Result] Top-5 Easiest: {sorted_classes[:5]}")
    print(f"[UNCG Result] Top-5 Hardest: {sorted_classes[-5:]}")

    # ====================================================
    # Step 3: 课程学习循环
    # ====================================================
    remaining_epochs = total_epochs - WARMUP_EPOCHS
    if remaining_epochs <= 0: remaining_epochs = 1
        
    curriculum_ratios = [0.4, 0.7, 1.0]
    phase_epochs = [
        int(remaining_epochs * 0.25),
        int(remaining_epochs * 0.25),
        remaining_epochs - int(remaining_epochs * 0.5)
    ]

    for phase_idx, (ratio, n_epochs) in enumerate(zip(curriculum_ratios, phase_epochs)):
        
        # A. 筛选类别
        num_keep = max(args.n_ways, int(args.num_base * ratio))
        active_classes = sorted_classes[:num_keep]
        active_classes_idx = np.sort(active_classes)
        
        print(f"\n====== Curriculum Phase {phase_idx+1}/3 | Keeping Top {int(ratio*100)}% ({len(active_classes)}) ======")

        # B. 构造 Dataset
        if args.dataset == 'FMC':
            curr_dataset = args.Dataset.FSDCLIPS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
        elif 'nsynth' in args.dataset:
            curr_dataset = args.Dataset.NDS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
        elif 'librispeech' in args.dataset:
            curr_dataset = args.Dataset.LBRS(root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
        else:
            curr_dataset = args.Dataset.S2S(dataset=args.dataset, root=args.dataroot, phase="train", index=active_classes_idx, base_sess=True, args=args)
        
        curr_loader = torch.utils.data.DataLoader(
            curr_dataset, 
            batch_size=args.dataloader.train_batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )

        # C. 训练
        for _ in range(n_epochs):
            # 【修正】移除多余参数
            model = standard_base_train(
                args, model, curr_loader, optimizer, scheduler, current_epoch
            )
            current_epoch += 1
            if scheduler is not None: scheduler.step()

    torch.save(dict(params=model.state_dict()), save_model_path)
    print(f"Base training finished. Model saved to {save_model_path}")
    return save_model_path
# def base_train(args, model):
#     # 1. 准备全量数据
#     # get_pretrain_dataloader 返回 (dataset, loader)
#     full_dataset, full_loader = get_pretrain_dataloader(args) 
#     save_model_path = os.path.join(args.save_dir, 'base_train_for_meta.pth')
    
#     # -----------------------------------------------------------
#     # Step 0: 快速热身 (Warm-up)
#     # -----------------------------------------------------------
#     # 先跑 5 个 Epoch，让模型参数从随机初始化状态稳定下来
#     # 否则计算出的"不确定性"是随机的，没有意义
#     print("\n=== [Step 0] Warm-up (5 Epochs) ===")
#     optimizer, scheduler = get_optimizer(model, args)
#     for epoch in range(5):
#         model = standard_base_train(args, model, full_loader, optimizer, scheduler, epoch)
        
#     # -----------------------------------------------------------
#     # Step 1: 难度评估 (Difficulty Assessment)
#     # -----------------------------------------------------------
#     sorted_classes = get_class_difficulty(args, model, full_loader)
#     print(f"\n[Curriculum] Easiest: {sorted_classes[:5]}")
#     print(f"[Curriculum] Hardest: {sorted_classes[-5:]}")
    
#     # -----------------------------------------------------------
#     # Step 2: 课程预训练 (Curriculum Pre-training)
#     # -----------------------------------------------------------
#     # 只训练容易和中等的类别，作为"优质初始化"
#     # 这里我们只跑两个短阶段，目的是调整 Feature Extractor 的权重分布
    
#     curriculum_phases = [0.5, 0.75] # 只训练前 50% 和 75%
#     phase_epochs = 5 # 每个阶段只跑 5 轮，快速过一遍
    
#     for idx, ratio in enumerate(curriculum_phases):
#         num_keep = int(args.num_base * ratio)
#         active_classes = sorted_classes[:num_keep]
#         active_classes = np.sort(active_classes) # 保持索引有序
        
#         print(f"\n=== [Step 2] Curriculum Phase {idx+1}: Top-{int(ratio*100)}% Easy Classes ===")
        
#         # 重新构建 Loader (只包含当前难度及以下的类别)
#         curr_dataset = args.Dataset.LBRS( # 请根据您的实际 Dataset 类名修改
#             root=args.dataroot, phase='train', index=active_classes, 
#             base_sess=True, args=args
#         )
#         curr_loader = torch.utils.data.DataLoader(
#             curr_dataset, batch_size=args.dataloader.train_batch_size, 
#             shuffle=True, num_workers=8, pin_memory=True
#         )
        
#         # 使用较小的学习率进行课程引导，防止破坏 Warmup 的成果
#         # 这里重置优化器，给一个固定的较小 LR
#         curr_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
#         for ep in range(phase_epochs):
#             model = standard_base_train(args, model, curr_loader, curr_optimizer, None, ep)

#     # -----------------------------------------------------------
#     # Step 3: 全量攻坚训练 (Full Training) - 关键修正！
#     # -----------------------------------------------------------
#     # 之前掉点就是因为这一步没跑够。
#     # 现在我们用 Curriculum 的参数作为起点，进行一次完整的、标准的训练。
#     # 这样既利用了课程学习带来的"良好初值"，又保证了困难样本被充分训练。
    
#     print(f"\n=== [Step 3] Full Standard Training ({args.epochs.epochs_std} Epochs) ===")
    
#     # 【关键】彻底重置优化器和调度器，使用满血学习率
#     final_optimizer, final_scheduler = get_optimizer(model, args)
    
#     best_acc = 0.0
#     for epoch in range(args.epochs.epochs_std):
#         # 使用全量 Loader
#         model = standard_base_train(args, model, full_loader, final_optimizer, final_scheduler, epoch)
        
#         # 记录最佳模型 (简单的 accuracy 监控)
#         # 这里为了不拖慢速度，直接保存最后一个，或者您可以加验证集测试
#         # ...
        
#         final_scheduler.step()

#     # 保存最终模型
#     torch.save(dict(params=model.state_dict()), save_model_path)
#     print(f"Base training finished. Model saved to {save_model_path}")
#     return save_model_path
def standard_base_train(args, model, trainloader, optimizer, scheduler, epoch):
    tl = Averager()
    ta = Averager()
    model = model.train()
    model.mode = 'encoder'
    
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        loss = F.cross_entropy(logits, train_label)
        
        acc = count_acc(logits, train_label)
        total_loss = loss

        # [核心修复]：判断 scheduler 是否存在
        # 如果是课程学习阶段(scheduler=None)，直接从 optimizer 获取学习率
        if scheduler is not None:
            lrc = scheduler.get_last_lr()[0]
        else:
            lrc = optimizer.param_groups[0]['lr']

        tqdm_gen.set_description(
                'Standard train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    tl = tl.item()
    ta = ta.item()
    print('ta:{},tl:{}'.format(ta,tl))
    return model
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
    cfg.update(vars(args))
    args = dict2namespace(cfg)
    set_seed(args.seed)  
    args.cuda = torch.cuda.is_available()
    check_randomness()
    train(args)
#切换数据集训练时，metatrainer132行和network49行要修改
