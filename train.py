import argparse
import yaml
import os
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
class LocalFeatureCluster(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256, k_ratio=0.3):
        super().__init__()
        self.k_ratio = k_ratio
        self.feat_dim = feat_dim
        
        # 局部特征处理器
        self.local_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 全局特征处理器
        self.global_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 动态融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, features):
        B, C, H, W = features.shape
        device = features.device  # 获取输入特征的设备
        
        # 确保所有子模块在正确设备上
        self.local_mlp = self.local_mlp.to(device)
        self.global_mlp = self.global_mlp.to(device)
        self.fusion_gate = self.fusion_gate.to(device)
        
        # 1. 提取全局特征
        global_feat = features.mean(dim=[2,3])  # [B,C]
        
        # 2. 局部聚类
        local_feat = features.view(B, C, -1).permute(0, 2, 1)  # [B,H*W,C]
        clustered_feat = []
        centers = []
        
        for i in range(B):
            # K-means聚类（确保数据在CPU）
            feat_np = local_feat[i].cpu().numpy()
            kmeans = KMeans(
                n_clusters=max(2, int(H*W*self.k_ratio)),
                n_init=10
            ).fit(feat_np)
            
            # 将结果移回原设备
            center = torch.from_numpy(kmeans.cluster_centers_).float().to(device)  # [k,C]
            label = torch.from_numpy(kmeans.labels_).to(device)  # [H*W]
            
            clustered_feat.append(center[label])  # [H*W,C]
            centers.append(center)
        
        clustered_feat = torch.stack(clustered_feat)  # [B,H*W,C]
        centers = torch.stack(centers)  # [B,k,C]
        
        # 3. MLP融合
        global_proj = self.global_mlp(global_feat).unsqueeze(1)  # [B,1,hidden]
        local_proj = self.local_mlp(clustered_feat)  # [B,H*W,hidden]
        
        # 动态权重
        gate_input = torch.cat([
            local_proj, 
            global_proj.expand(-1, H*W, -1)
        ], dim=-1)  # [B,H*W,hidden*2]
        
        fusion_weight = self.fusion_gate(gate_input)  # [B,H*W,C]
        
        # 4. 特征合成
        enhanced_feat = (fusion_weight * clustered_feat + 
                        (1 - fusion_weight) * local_feat)
        
        # 恢复空间结构
        enhanced_feat = enhanced_feat.permute(0, 2, 1).view(B, C, H, W)
        
        return enhanced_feat, centers
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
    
    # 环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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
    parser.add_argument('--dataroot', type=str,default="/data/datasets/librispeech_fscil/")
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--save_result',type = str,default='/data/lqq/baseline/save_result/')
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=80, type=int)
    parser.add_argument('--checkpoint', type=bool, default=True)
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

def plot_tsne_comparison(original_features, enhanced_features, labels, save_path, timestamp):
    """
    Plot t-SNE visualization for both original and enhanced features with unified scales.
    """
    # Compute t-SNE for original features
    tsne_original = TSNE(n_components=2, perplexity=min(30, len(original_features)-1), random_state=42)
    original_2d = tsne_original.fit_transform(original_features)
    
    # Compute t-SNE for enhanced features
    tsne_enhanced = TSNE(n_components=2, perplexity=min(30, len(enhanced_features)-1), random_state=42)
    enhanced_2d = tsne_enhanced.fit_transform(enhanced_features)
    
    # Determine unified axis limits
    all_x = np.concatenate([original_2d[:, 0], enhanced_2d[:, 0]])
    all_y = np.concatenate([original_2d[:, 1], enhanced_2d[:, 1]])
    x_min, x_max = all_x.min() - 5, all_x.max() + 5
    y_min, y_max = all_y.min() - 5, all_y.max() + 5
    
    # Create side-by-side plots
    plt.figure(figsize=(14, 6))
    
    # Original features plot
    plt.subplot(121)
    plt.scatter(original_2d[:, 0], original_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title('Original Features t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.colorbar(label='Class Labels')
    
    # Enhanced features plot
    plt.subplot(122)
    plt.scatter(enhanced_2d[:, 0], enhanced_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title('Enhanced Features t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.colorbar(label='Class Labels')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'tsne_comparison_{timestamp}.png'), dpi=300)
    plt.close()

def plot_correlation_heatmaps(original_features, enhanced_features, save_path, timestamp):
    """
    Plot correlation heatmaps with numerical annotations for both original and enhanced features.
    """
    plt.figure(figsize=(14, 6))
    
    # Original features correlation
    plt.subplot(121)
    corr_original = np.corrcoef(original_features[:20])  # Top 20 samples for clarity
    sns.heatmap(corr_original, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6})
    plt.title('Original Features Correlation Matrix')
    
    # Enhanced features correlation
    plt.subplot(122)
    corr_enhanced = np.corrcoef(enhanced_features[:20])
    sns.heatmap(corr_enhanced, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6})
    plt.title('Enhanced Features Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'correlation_heatmaps_{timestamp}.png'), dpi=300)
    plt.close()

def plot_feature_distribution(original_features, enhanced_features, save_path, timestamp):
    """
    Plot overlaid feature value distributions for comparison.
    """
    plt.figure(figsize=(10, 6))
    
    # Compute statistics
    orig_mean = np.mean(original_features)
    orig_std = np.std(original_features)
    enh_mean = np.mean(enhanced_features)
    enh_std = np.std(enhanced_features)
    
    # Plot overlaid histograms
    plt.hist(original_features.flatten(), bins=50, color='blue', alpha=0.5, label=f'Original (μ={orig_mean:.2f}, σ={orig_std:.2f})')
    plt.hist(enhanced_features.flatten(), bins=50, color='orange', alpha=0.5, label=f'Enhanced (μ={enh_mean:.2f}, σ={enh_std:.2f})')
    
    plt.title('Feature Value Distribution Comparison')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'feature_distribution_{timestamp}.png'), dpi=300)
    plt.close()

def calculate_metrics(original_features, enhanced_features, labels, save_path, timestamp):
    """
    Calculate and save clustering metrics for both original and enhanced features.
    """
    metrics = {}
    
    # Silhouette Score (higher is better)
    if len(np.unique(labels)) > 1:  # Check if there's more than one cluster
        sil_orig = silhouette_score(original_features, labels)
        sil_enh = silhouette_score(enhanced_features, labels)
        metrics['silhouette_score'] = {'original': sil_orig, 'enhanced': sil_enh}
    
    # Intra/inter class distance ratio (higher is better)
    def intra_inter_ratio(features, labels):
        intra_dists = []
        inter_dists = []
        unique_labels = np.unique(labels)
        
        for i in range(len(unique_labels)):
            mask_i = (labels == unique_labels[i])
            # Intra-class distances
            intra_dist = pairwise_distances(features[mask_i]).mean()
            intra_dists.append(intra_dist)
            
            # Inter-class distances
            for j in range(i+1, len(unique_labels)):
                mask_j = (labels == unique_labels[j])
                inter_dist = pairwise_distances(features[mask_i], features[mask_j]).mean()
                inter_dists.append(inter_dist)
        
        return np.mean(inter_dists) / np.mean(intra_dists) if np.mean(intra_dists) > 0 else 0
    
    ratio_orig = intra_inter_ratio(original_features, labels)
    ratio_enh = intra_inter_ratio(enhanced_features, labels)
    metrics['inter_intra_ratio'] = {'original': ratio_orig, 'enhanced': ratio_enh}
    
    # Save metrics to text file
    with open(os.path.join(save_path, f'clustering_metrics_{timestamp}.txt'), 'w') as f:
        f.write("Clustering Metrics Comparison:\n")
        f.write("="*40 + "\n")
        for metric, values in metrics.items():
            f.write(f"{metric}:\n")
            for feature_type, value in values.items():
                f.write(f"  - {feature_type}: {value:.4f}\n")
            f.write("\n")
    
    return metrics
def visualize_decision_boundary(model, features, labels, save_path, title, enhanced=False):
    """展示决策边界变化（基于特征向量，避免调用model.encode）"""
    # 仅保留前2维特征用于可视化（但不输入模型）
    features_2d = features[:, :2]  # 用于绘图的2D坐标
    model.eval()
    
    # 生成网格数据（基于2D特征的坐标范围）
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # 网格点的2D坐标

    # 关键修改：直接使用特征向量计算与原型的距离，不调用model.encode
    with torch.no_grad():
        # 获取模型的原型（分类头权重）
        proto = model.fc.weight.detach().cpu().numpy()  # [num_classes, feat_dim]
        
        # 为了计算网格点的分类，需要将2D网格点映射回高维特征空间
        # 这里简化处理：假设网格点的高维特征是其2D坐标+其余维度为0（仅用于可视化）
        # 注意：这是近似处理，实际应根据特征分布插值，此处为了规避维度错误
        feat_dim = features.shape[1]  # 原始特征维度（如512）
        grid_feat = np.zeros((len(grid_points), feat_dim))
        grid_feat[:, :2] = grid_points  # 前两维用网格坐标，其余为0
        
        # 计算与原型的余弦相似度
        grid_feat_tensor = torch.FloatTensor(grid_feat).cuda()
        proto_tensor = torch.FloatTensor(proto).cuda()
        logits = F.cosine_similarity(grid_feat_tensor.unsqueeze(1), proto_tensor, dim=-1)
        Z = logits.argmax(1).cpu().numpy()
    
    # 绘制决策边界
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, s=20, edgecolor='k')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
def debug_cluster(args, model, data, labels, session=None):
    """
    Enhanced clustering with comprehensive feature visualization and metric calculation.
    """
    # Generate unique timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    test_round = args.current_test if hasattr(args, 'current_test') else 0
    
    # Extract features
    with torch.no_grad():
        features = torch.stack([model.hgnn_encode(x).squeeze() for x in data])
        # 将原始特征从 GPU 移动到 CPU 并转换为 NumPy 数组
        original_features = features.mean(dim=[2,3]).cpu().numpy()  # [N, 512]
        
        # Feature enhancement
        global_feat = features.mean(dim=[2,3])  # [N, 512]，此时在 GPU 上
        clustered_feat, _ = LocalFeatureCluster(k_ratio=0.5 + session*0.1)(features)
        local_feat = clustered_feat.mean(dim=[2,3])  # [N, 512]，在 GPU 上
        
        alpha = min(0.7, 0.5 + session*0.1)
        # 运算前确保张量在同一设备（这里都是 GPU 张量，可直接运算）
        enhanced_feat = alpha * global_feat + (1 - alpha) * local_feat  # [N, 512]，结果在 GPU 上
        # 将增强后的特征移动到 CPU 并转换为 NumPy 数组
        enhanced_features = enhanced_feat.cpu().numpy()
    visualize_decision_boundary(
    model, original_features, labels, 
    os.path.join(args.save_result, f'original_boundary_{timestamp}.png'), 
    'Original Features Decision Boundary'
)
    visualize_decision_boundary(
    model, enhanced_features, labels, 
    os.path.join(args.save_result, f'enhanced_boundary_{timestamp}.png'), 
    'Enhanced Features Decision Boundary'
)
    # 计算增强前后特征差异时，要基于 CPU 上的张量或数组
    # 先把 enhanced_feat 移动到 CPU 再转换为 NumPy 数组参与运算
    enhanced_feat_cpu = enhanced_feat.cpu()
    original_features_tensor = torch.from_numpy(original_features).to(enhanced_feat_cpu.device)  # 把原始特征数组转回张量并放到同一设备
    diff = torch.mean(torch.abs(enhanced_feat_cpu - original_features_tensor)).item()
    print("增强前后特征差异:", diff)  # 应>0
    
    # Create save directory
    base_dir = os.path.join(args.save_result, f"test_{test_round}_session_{session}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate visualizations
    plot_tsne_comparison(original_features, enhanced_features, labels, base_dir, timestamp)
    plot_correlation_heatmaps(original_features, enhanced_features, base_dir, timestamp)
    plot_feature_distribution(original_features, enhanced_features, base_dir, timestamp)
    
    # Calculate and save metrics
    metrics = calculate_metrics(original_features, enhanced_features, labels, base_dir, timestamp)
    
    # Print metrics to console
    print(f"\n=== Clustering Metrics (Session {session}, Test {test_round}) ===")
    for metric, values in metrics.items():
        print(f"{metric}:")
        for feature_type, value in values.items():
            print(f"  - {feature_type}: {value:.4f}")
        improvement = (values['enhanced'] - values['original']) / values['original'] * 100
        print(f"  => Improvement: {improvement:.2f}%")
    print("="*40)
    
    # Perform clustering
    kmeans = KMeans(
        n_clusters=args.num_unlabeled_classes,
        n_init=20,
        random_state=args.seed
    ).fit(enhanced_features)
    y = kmeans.labels_
    
    # Calculate clustering accuracy
    labels = np.array(labels)
    acc, map_result = cluster_acc(args, labels, y)
    
    # Update model prototypes
    updated = 0
    for cluster_id in np.unique(y):
        if cluster_id in map_result and map_result[cluster_id] >= args.num_labeled_classes:
            indices = np.where(y == cluster_id)[0]
            if len(indices) > 0:
                with torch.no_grad():
                    # 注意这里要使用 CPU 上的增强特征张量来更新（如果模型在 GPU 上，后续会自动处理设备迁移）
                    model.fc.weight.data[map_result[cluster_id]] = enhanced_feat[indices].mean(dim=0).cpu()
                    updated += 1
    
    print(f"Session {session} (Test {test_round}, {timestamp}): Acc={acc:.4f}, Updated={updated} classes")
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
        for j in range(0, args.num_session): 
            result['sess{}_ak'.format(j)]=[]
            result['sess{}_au'.format(j)]=[]
            result['sess{}_fs'.format(j)]=[]
            result['sess{}_inc'.format(j)]=[]
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
            result['sess0_inc'].append(base_acc)
            # 打印session 0结果
            print(f"Session 0: acc known: {base_acc:.4f}, acc unknown: 0.0000, "
                  f"f1 score: 0.0000, incremental acc: {base_acc:.4f}")
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
                inc_acc=test(args, model, testloader,  session)
                result['sess{}_inc'.format(session)]+=[inc_acc]
                args.num_labeled_classes += args.way
                avg_acc_known = sum(result['sess{}_ak'.format(session)]) / len(result['sess{}_ak'.format(session)])  
                avg_acc_unknown = sum(result['sess{}_au'.format(session)]) / len(result['sess{}_au'.format(session)])  
                avg_fscore = sum(result['sess{}_fs'.format(session)]) / len(result['sess{}_fs'.format(session)])  
                avg_inc_acc = sum(result['sess{}_inc'.format(session)]) / len(result['sess{}_inc'.format(session)])  
                session_ka[i].append(avg_acc_known)
                session_uka[i].append(avg_acc_unknown)
                session_f1s[i].append(avg_fscore)
                session_inc[i].append(avg_inc_acc)
                # avg_session0_acc = sum(session0_acc_list) / len(session0_acc_list)
                # print(f"\n=== Final Average Session 0 Acc: {avg_session0_acc:.4f} ===")
                # result_file.write(f"\nAverage Session 0 Acc: {avg_session0_acc:.4f}\n")
                # 写入文件  
                result_line = "session: {}, aac known: {:.4f}, acc unknown: {:.4f}, f1 score: {:.4f}, incremental acc: {:.4f}\n".format(  
                    session, avg_acc_known, avg_acc_unknown, avg_fscore, avg_inc_acc)  
                result_file.write(result_line)  
                print("session:{},acc known:{:.4f},acc unknown:{:.4f},f1 score:{:.4f},incremental acc:{:.4f}".format(session,(sum(result['sess{}_ak'.format(session)])/len(result['sess{}_ak'.format(session)])), 
           (sum(result['sess{}_au'.format(session)])/len(result['sess{}_au'.format(session)])),(sum(result['sess{}_fs'.format(session)])/len(result['sess{}_fs'.format(session)])), sum(result['sess{}_inc'.format(session)])/len(result['sess{}_inc'.format(session)])))
            best_model_dir = os.path.join(args.save_dir, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=model.state_dict()), best_model_dir)
        session0_acc_values = np.array(session0_acc_list)
        session0_mean = np.mean(session0_acc_values)
        session0_std = np.std(session0_acc_values)
        
        print(f"\n=== Final Session 0 ===")
        print(f"Average Acc: {session0_mean:.4f} ± {session0_std:.4f}")
        result_file.write(f"\n=== Final Session 0 ===\n")
        result_file.write(f"Average Acc: {session0_mean:.4f} ± {session0_std:.4f}\n")
        for ses in range(args.num_session-1):
            # 计算均值和标准差
            ka_values = [session_ka[time][ses] for time in range(args.test_times)]
            uka_values = [session_uka[time][ses] for time in range(args.test_times)]
            f1s_values = [session_f1s[time][ses] for time in range(args.test_times)]
            inc_values = [session_inc[time][ses] for time in range(args.test_times)]
            
            ka_mean = np.mean(ka_values)
            ka_std = np.std(ka_values)
            uka_mean = np.mean(uka_values)
            uka_std = np.std(uka_values)
            f1s_mean = np.mean(f1s_values)
            f1s_std = np.std(f1s_values)
            inc_mean = np.mean(inc_values)
            inc_std = np.std(inc_values)

            # 打印带标准差的结果（保持原有打印格式）
            print(f"total session{ses+1} acc known is {ka_mean:.4f} ± {ka_std:.4f}")
            print(f"total session{ses+1} acc unknown is {uka_mean:.4f} ± {uka_std:.4f}")
            print(f"total session{ses+1} f1 score is {f1s_mean:.4f} ± {f1s_std:.4f}")
            print(f"total session{ses+1} incremental acc is {inc_mean:.4f} ± {inc_std:.4f}")
            
            # 写入文件（保持原有格式）
            result_row = (
                f"session: {ses+1}, "
                f"total aac known: {ka_mean:.4f} ± {ka_std:.4f}, "
                f"total acc unknown: {uka_mean:.4f} ± {uka_std:.4f}, "
                f"total f1 score: {f1s_mean:.4f} ± {f1s_std:.4f}, "
                f"total incremental acc: {inc_mean:.4f} ± {inc_std:.4f}\n"
            )
            result_file.write(result_row)  
                

def base_train(args,model):
    data_dict = {}
    data_dict['train_set'],data_dict['trainloader']= get_pretrain_dataloader(args) 
    net_dict = {}
    
    net_dict['optimizer'], net_dict['scheduler'] = get_optimizer(model, args)
    save_model_path = os.path.join(args.save_dir, f'base_train_for_meta.pth')
    #encoder pretrain
    
    for epoch in range(args.epochs.epochs_std):
        model=standard_base_train(args, model,data_dict['trainloader'],net_dict['optimizer'], net_dict['scheduler'], epoch) #要不要打印，改对了吗
        net_dict['epoch'] = epoch
        net_dict['scheduler'].step()
    torch.save(dict(params=model.state_dict()), save_model_path)
   
    return save_model_path

def standard_base_train(args, model, trainloader, optimizer, scheduler, epoch):
    num_base = args.num_base
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
        # feat, proj_feat = model.encode(data, return_proj=True)
        # contrast_loss = model.compute_contrastive_loss(proj_feat[:len(proj_feat)//2], 
        #                                              proj_feat[len(proj_feat)//2:])
        # loss = F.cross_entropy(model.fc(feat), train_label)
        acc = count_acc(logits, train_label)
        total_loss = loss
        # total_loss = loss

        lrc = scheduler.get_last_lr()[0]
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
#其他聚类方式
# def visualize_feature_analysis(features, labels, save_dir):
#     """Feature space analysis"""
#     plt.figure(figsize=(18, 6))
    
#     # 1. PCA projection
#     pca = PCA(n_components=2)
#     pca_feats = pca.fit_transform(features)
    
#     # 2. Distance distribution
#     plt.subplot(131)
#     dists = pairwise_distances(features).flatten()
#     plt.hist(dists[dists > 0], bins=50)
#     plt.title(f"Distance Distribution (median={np.median(dists):.2f})")
    
#     # 3. PCA visualization
#     plt.subplot(132)
#     plt.scatter(pca_feats[:,0], pca_feats[:,1], alpha=0.6)
#     plt.title("PCA Projection")
    
#     # 4. Similarity matrix
#     plt.subplot(133)
#     sns.heatmap(cosine_similarity(features[:20]), cmap="YlGnBu", annot=True, fmt=".2f")
#     plt.title("Top20 Sample Similarity")
    
#     plt.savefig(f"{save_dir}/feature_analysis.png")
#     plt.close()
# def visualize_cluster_results(features, y, true_labels, save_dir):
#     """Cluster visualization"""
#     try:
#         plt.figure(figsize=(12, 5))
        
#         pca = PCA(n_components=2)
#         viz_feats = pca.fit_transform(features)
        
#         # 1. Cluster results
#         plt.subplot(121)
#         unique_labels = np.unique(y)
#         for lbl in unique_labels:
#             mask = (y == lbl)
#             plt.scatter(viz_feats[mask, 0], viz_feats[mask, 1], 
#                        label=f'Cluster {lbl}', alpha=0.6)
#         plt.title(f"Cluster Results (K={len(unique_labels)})")
        
#         # 2. True labels if available
#         if true_labels is not None:
#             plt.subplot(122)
#             for lbl in np.unique(true_labels):
#                 mask = (np.array(true_labels) == lbl)
#                 plt.scatter(viz_feats[mask, 0], viz_feats[mask, 1], 
#                            label=f'Class {lbl}', alpha=0.6)
#             plt.title("True Labels")
        
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.tight_layout()
#         plt.savefig(f"{save_dir}/cluster_results.png", dpi=300)
#         plt.close()
        
#     except Exception as e:
#         print(f"Visualization failed: {str(e)}")
# def debug_cluster(args, model, data, labels, session):
#     """Enhanced clustering with diagnostics"""
#     # Feature extraction
#     with torch.no_grad():
#         features = torch.stack([model.encode(x).squeeze() for x in data]).cpu().numpy()
    
#     # Feature normalization
#     features = robust_feature_scaling(features)
#     labels = np.array(labels) if labels is not None else None
    
#     # Feature visualization
#     if args.debug:
#         visualize_feature_analysis(features, labels, args.save_result)
    
#     # Dynamic clustering parameters
#     min_samples = max(3, len(features)//10)
#     init_radius = 0.5 if len(features) < 100 else 0.3
    
#     # Initialize clusterer
#     global clusterer
#     if 'clusterer' not in globals():
#         clusterer = FStream(
#             init_radius=init_radius,
#             min_weight=1.0/len(features),
#             k=min(5, len(features)-1)
#         )
    
#     # Cluster with fallback
#     try:
#         clusterer.process_batch(features)
#         _, y = clusterer.get_clusters()
        
#         # Fallback to KMeans if clustering failed
#         if len(np.unique(y)) <= 1:
#             print("Fallback to KMeans clustering")
#             y = KMeans(n_clusters=args.num_unlabeled_classes, n_init=10).fit_predict(features)
#     except Exception as e:
#         print(f"Clustering error: {str(e)}")
#         y = np.zeros(len(features))  # Default to single cluster
    
#     # Visualization
#     if args.debug:
#         visualize_cluster_results(features, y, labels, args.save_result)
    
#     return update_model_safely(args, model, data, y, labels)
    
#     # 流式处理
#     clusterer.process_batch(features)
#     cluster_centers, y = clusterer.get_clusters()
#     # 确保聚类结果与输入长度一致
#     if len(y) != len(features):
#         y = y[:len(features)]
#     # 结果验证与可视化
#     if args.debug:
#         safe_visualization(features, y, labels, args.save_result)
    
#     return update_model_safely(args, model, data, y, labels)

# def kmeans_visualization(features, labels, save_dir):
#     """仅用于诊断的K-means可视化"""
#     from sklearn.cluster import KMeans
#     try:
#         # 使用真实类别数作为K值
#         true_k = len(np.unique(labels)) if labels is not None else args.num_unlabeled_classes
#         kmeans = KMeans(n_clusters=min(true_k, len(features)), n_init=10)
#         pred_labels = kmeans.fit_predict(features)
        
#         # PCA降维可视化
#         pca = PCA(n_components=2)
#         viz_feats = pca.fit_transform(features)
        
#         plt.figure(figsize=(12, 5))
#         plt.subplot(121)
#         plt.scatter(viz_feats[:,0], viz_feats[:,1], c=pred_labels, cmap='tab20')
#         plt.title("K-means Diagnostic (Colored by Cluster)")
        
#         if labels is not None:
#             plt.subplot(122)
#             plt.scatter(viz_feats[:,0], viz_feats[:,1], c=labels, cmap='tab20')
#             plt.title("True Labels")
        
#         plt.savefig(f"{save_dir}/kmeans_diagnostic.png")
#         plt.close()
        
#         # 打印K-means评估指标
#         from sklearn.metrics import silhouette_score
#         sil_score = silhouette_score(features, pred_labels)
#         print(f"K-means诊断: silhouette_score={sil_score:.3f}")
        
#     except Exception as e:
#         print(f"K-means诊断可视化失败: {str(e)}")
# def run_optimal_cluster(features, n_classes=5, session=None):
#     """
#     动态混合聚类策略（根据session阶段自动选择最优方法）
#     参数：
#         features: 输入特征矩阵
#         n_classes: 目标类别数
#         session: 当前session编号（用于动态调整）
#     返回：
#         聚类标签数组
#     """
#     from sklearn.cluster import KMeans, SpectralClustering
#     from sklearn.mixture import GaussianMixture
#     from sklearn.preprocessing import StandardScaler
    
#     # 1. 特征标准化
#     features = StandardScaler().fit_transform(features)
#     n_samples = len(features)
    
#     # 2. 动态策略选择
#     if n_samples < 15:  # 极少量样本
#         method = KMeans(n_clusters=min(n_classes, n_samples), n_init=10)
#     elif session is not None and session >= 3:  # 后期session
#         method = GaussianMixture(n_components=n_classes, covariance_type='spherical')
#     else:  # 常规情况
#         method = SpectralClustering(
#             n_clusters=n_classes,
#             affinity='nearest_neighbors',
#             n_neighbors=min(5, n_samples-1),
#             assign_labels='kmeans'
#         )
    
#     # 3. 执行聚类
#     try:
#         if isinstance(method, GaussianMixture):
#             return method.fit_predict(features)
#         else:
#             return method.fit_predict(features)
#     except:
#         return KMeans(n_clusters=min(n_classes, n_samples)).fit_predict(features)

# def update_model_safely(args, model, data, y, labels):
#     """Safe model updating with clustering results"""
#     try:
#         y = np.array(y)
#         labels = np.array(labels)
        
#         # Validate clustering results
#         if len(y) != len(labels):
#             y = y[:len(labels)]
        
#         # Remove negative labels
#         if (y < 0).any():
#             print(f"Warning: Negative cluster labels found, setting to 0")
#             y = np.maximum(y, 0)
        
#         # Calculate clustering accuracy
#         acc, map = cluster_acc(args, labels, y)
#         print(f"Clustering accuracy: {acc:.4f}, Mapping: {map}")
        
#         # Update model weights
#         updated = 0
#         with torch.no_grad():
#             for cluster_id in np.unique(y):
#                 if cluster_id in map:
#                     true_label = map[cluster_id]
#                     if true_label >= args.num_labeled_classes:
#                         indices = np.where(y == cluster_id)[0]
#                         if len(indices) > 0:
#                             feats = torch.stack([model.encode(data[i]).detach().clone() 
#                                               for i in indices if i < len(data)])
#                             if len(feats) > 0:
#                                 model.fc.weight.data[true_label] = feats.mean(dim=0)
#                                 updated += 1
#                                 print(f"Updated class {true_label} with {len(indices)} samples")
        
#         print(f"Successfully updated {updated} classes")
#         return acc if acc > 0 else 0.0  # Ensure non-negative accuracy
    
#     except Exception as e:
#         print(f"Model update error: {str(e)}")
#         return 0.0

# def safe_visualization(features, y, labels, save_dir):
#     """增强版可视化（修复维度不匹配问题）"""
#     try:
#         # 确保所有数组长度一致
#         min_length = min(len(features), len(y))
#         if labels is not None:
#             min_length = min(min_length, len(labels))
        
#         features = features[:min_length]
#         y = y[:min_length]
#         if labels is not None:
#             labels = labels[:min_length]
        
#         # 使用TSNE降维
#         viz_feats = TSNE(n_components=2, perplexity=min(30, len(features)-1)).fit_transform(features)
        
#         plt.figure(figsize=(18,6))
        
#         # 1. 流聚类结果
#         plt.subplot(131)
#         unique_labels = np.unique(y)
#         for lbl in unique_labels:
#             mask = (y == lbl)
#             plt.scatter(viz_feats[mask,0], viz_feats[mask,1], label=f'Cluster {lbl}', alpha=0.6)
#         plt.title(f"FStream (K={len(unique_labels)})")
        
#         # 2. K-means对比
#         plt.subplot(132)
#         km_labels = KMeans(n_clusters=len(unique_labels), n_init=10).fit_predict(features)
#         plt.scatter(viz_feats[:,0], viz_feats[:,1], c=km_labels, cmap='tab20')
#         plt.title("K-means对比")
        
#         # 3. 真实标签（如果可用）
#         if labels is not None:
#             plt.subplot(133)
#             plt.scatter(viz_feats[:,0], viz_feats[:,1], c=labels[:len(viz_feats)], cmap='tab20')
#             plt.title("True Labels")
        
#         plt.savefig(f"{save_dir}/stream_cluster_compare.png", dpi=300)
#         plt.close()
        
#     except Exception as e:
#         print(f"可视化异常: {str(e)}")
#         # 打印调试信息
#         print(f"Debug info - Features shape: {features.shape if hasattr(features, 'shape') else len(features)}")
#         print(f"Debug info - y length: {len(y)}")
#         if labels is not None:
#             print(f"Debug info - labels length: {len(labels)}")
# def robust_feature_scaling(features):
#     """增强版特征标准化"""
#     from sklearn.pipeline import make_pipeline
#     from sklearn.preprocessing import RobustScaler, PowerTransformer
    
#     features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
#     # 两阶段标准化
#     pipeline = make_pipeline(
#         RobustScaler(quantile_range=(5, 95)),
#         PowerTransformer(method='yeo-johnson')
#     )
#     return pipeline.fit_transform(features)