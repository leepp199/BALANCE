import argparse
import yaml
import os
import torch
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

class LocalFeatureCluster(nn.Module):
    def __init__(self, k_ratio=0.5):
        super().__init__()
        self.k_ratio = k_ratio  # 簇数量比例（默认取特征数的1/4）

    def forward(self, features):
        """
        输入: 
            features: [B, C, H, W]  # B=批大小, C=通道数, H/W=特征图高宽
        输出:
            clustered_features: [B, C, H, W]  # 聚类增强后的特征
            centroids: [B, k, C]    # 各样本的簇中心
        """
        B, C, H, W = features.shape
        num_features = H * W
        k = max(1, int(num_features * self.k_ratio))  # 计算簇数量
        
        # 1. 展开特征为局部向量 [B, H*W, C]
        local_features = features.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        centroids = []
        clustered_features = []
        
        for i in range(B):
            # 2. 对每个样本单独聚类（避免样本间干扰）
            kmeans = KMeans(n_clusters=k, n_init=10,random_state=args.seed)
            kmeans.fit(local_features[i].cpu().numpy())  # [H*W, C]
            
            # 3. 获取簇中心和标签
            centroid = torch.from_numpy(kmeans.cluster_centers_).float().to(features.device)  # [k, C]
            labels = torch.from_numpy(kmeans.labels_).to(features.device)  # [H*W]
            
            # 4. 将簇中心拼接到原始特征（论文公式8）
            expanded_centroid = centroid[labels]  # [H*W, C]
            enhanced_feature = local_features[i] + expanded_centroid  # 特征增强
            
            centroids.append(centroid)
            clustered_features.append(enhanced_feature)
        
        # 重组为原始形状 [B, C, H, W]
        clustered_features = torch.stack(clustered_features).permute(0, 2, 1).view(B, C, H, W)
        centroids = torch.stack(centroids)  # [B, k, C]
        
        return clustered_features, centroids

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
    # 添加深度聚类相关参数
    parser.add_argument('--enable_cluster', action='store_true', help='Enable deep clustering')
    parser.add_argument('--cluster_epochs', type=int, default=20, help='Number of clustering epochs')
    parser.add_argument('--num_clusters', type=int, default=10, help='Number of clusters')
    parser.add_argument('--lambda_cluster', type=float, default=0.5, help='Weight for cluster loss')
    parser.add_argument('--cluster_lr', type=float, default=1e-4, help='Learning rate for clustering')
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
def debug_cluster(args, model, data, labels, session=None):
    """改进版层次化聚类方案B++ (带可视化)"""
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join(args.save_result, 
                         f"session_{session}_test_{args.current_test}_time_{timestamp}")
    os.makedirs(vis_dir, exist_ok=True)
    # 1. 特征提取与增强
    with torch.no_grad():
        # 获取原始特征 [N,512,13,4]
        features = torch.stack([model.hgnn_encode(x).squeeze() for x in data])
        
        # ========== 特征可视化1: 原始特征分布 ==========
        plt.figure(figsize=(12, 5))
        # 随机选择5个通道展示分布
        for i in np.random.choice(range(features.shape[1]), 5, replace=False):
            sns.kdeplot(features[:, i, :, :].flatten().cpu().numpy(), label=f'Channel {i}')
        plt.title(f"Session {session} - Original Feature Distribution")
        plt.legend()
        plt.savefig(os.path.join(vis_dir, "1_original_feature_dist.png"))
        plt.close()
        
        # 全局平均特征 [N,512]
        global_feat = features.mean(dim=[2,3])  
        
        # 局部聚类增强（增大k_ratio保留更多细节）
        cluster = LocalFeatureCluster(k_ratio=0.5 + session*0.1)  # 动态调整k_ratio
        clustered_feat, centroids = cluster(features)  # [N,512,13,4], [N,k,512]
        local_feat = clustered_feat.mean(dim=[2,3])  # [N,512]
        
        # ========== 特征可视化2: 聚类前后特征对比 ==========
        sample_idx = np.random.randint(0, len(data))
        plt.figure(figsize=(15, 6))
        plt.subplot(121)
        plt.imshow(features[sample_idx, :16].mean(0).cpu().numpy(), cmap='viridis')
        plt.title(f"Sample {sample_idx} Original Features")
        plt.subplot(122)
        plt.imshow(clustered_feat[sample_idx, :16].mean(0).cpu().numpy(), cmap='viridis')
        plt.title(f"Sample {sample_idx} Clustered Features")
        plt.savefig(os.path.join(vis_dir, "2_feature_comparison.png"))
        plt.close()
        
        # 动态加权融合特征 [N,512]
        alpha = min(0.7, 0.5 + session*0.1)  # 随session增加局部特征权重
        enhanced_feat = alpha * global_feat + (1-alpha) * local_feat
        
        # ========== 特征可视化3: 特征融合效果 ==========
        plt.figure(figsize=(12, 5))
        plt.scatter(global_feat.mean(1).cpu(), local_feat.mean(1).cpu(), alpha=0.6)
        plt.xlabel("Global Feature Magnitude")
        plt.ylabel("Local Feature Magnitude")
        plt.title(f"Feature Fusion (alpha={alpha:.2f})")
        plt.savefig(os.path.join(vis_dir, "3_feature_fusion.png"))
        plt.close()
    
    # 2. 全局聚类（固定簇数）
    all_centroids = centroids.view(-1, centroids.shape[-1]).cpu().numpy()
    
    # ========== 特征可视化4: 簇中心分布 ==========
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.corrcoef(all_centroids.T), cmap='coolwarm', center=0)
    plt.title("Centroid Correlation Matrix")
    plt.savefig(os.path.join(vis_dir, "4_centroid_correlation.png"))
    plt.close()
    
    global_kmeans = KMeans(
        n_clusters=args.num_unlabeled_classes,
        n_init=20,
        random_state=args.seed
    ).fit(all_centroids)
    
    # 3. 置信度加权标签分配
    y = []
    for i in range(len(data)):
        # 计算每个局部中心的权重（距离倒数）
        dists = pairwise_distances(
            centroids[i].cpu().numpy(),
            global_kmeans.cluster_centers_,
            metric='cosine'
        )
        weights = 1 / (dists + 1e-6)
        weighted_vote = np.argmax(weights.sum(axis=0))
        y.append(weighted_vote)
    y = np.array(y)
    
    # 关键修复点：确保接收cluster_acc返回的两个值
    acc, map = cluster_acc(args, np.array(labels), y)
    
    # ========== 特征可视化5: 聚类结果 ==========
    try:
        # 动态设置perplexity（不超过样本数-1）
        perplexity = min(30, len(enhanced_feat)-1) if len(enhanced_feat) > 1 else 1
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=args.seed
        )
        vis_feats = tsne.fit_transform(enhanced_feat.cpu().numpy())

        plt.figure(figsize=(18, 6))

        # 子图1：映射后的聚类结果
        plt.subplot(131)
        # 使用正确的map变量（已从cluster_acc获取）
        mapped_y = np.array([map.get(cluster_id, -1) for cluster_id in y])  # -1表示未映射的簇
        unique_mapped = np.unique(mapped_y)
        for lbl in unique_mapped:
            if lbl != -1:  # 过滤未映射的簇
                mask = (mapped_y == lbl)
                plt.scatter(vis_feats[mask, 0], vis_feats[mask, 1], 
                       label=f'Mapped Cluster {lbl}', alpha=0.6)
        plt.title("Mapped Cluster Assignment")

        # 子图2：真实标签
        plt.subplot(132)
        if labels is not None:
            for lbl in np.unique(labels):
                mask = (np.array(labels) == lbl)
                plt.scatter(vis_feats[mask, 0], vis_feats[mask, 1], 
                       label=f'True {lbl}', alpha=0.6)
            plt.title("True Labels")

        # 子图3：正确/错误分类
        plt.subplot(133)
        if labels is not None:
            correct = (mapped_y == np.array(labels))
            plt.scatter(vis_feats[correct, 0], vis_feats[correct, 1], 
                   c='green', label='Correct')
            plt.scatter(vis_feats[~correct, 0], vis_feats[~correct, 1], 
                   c='red', label='Wrong')
            plt.legend()
            plt.title(f"Accuracy: {acc:.2%}")

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "5_cluster_results.png"))
        plt.close()
        
    except Exception as e:
        print(f"t-SNE可视化失败: {str(e)}")
        # 改用PCA降维
        pca = PCA(n_components=2)
        vis_feats = pca.fit_transform(enhanced_feat.cpu().numpy())
        
        plt.figure(figsize=(12, 6))
        plt.scatter(vis_feats[:, 0], vis_feats[:, 1], c=y, cmap='tab20', alpha=0.6)
        plt.title("PCA Visualization (t-SNE failed)")
        plt.savefig(os.path.join(vis_dir, "5_cluster_results.png"))
        plt.close()
    
    # 4. 弹性原型更新
    updated = 0
    for cluster_id in np.unique(y):
        if cluster_id in map:
            true_label = map[cluster_id]
            if true_label >= args.num_labeled_classes:
                indices = np.where(y == cluster_id)[0]
                if len(indices) > 0:
                    with torch.no_grad():
                        # 计算新原型
                        new_proto = torch.stack(
                            [model.encode(data[i]) for i in indices]
                        ).mean(dim=0)
                        
                        # 动量更新
                        momentum = 0.7 if session > 0 else 0.3
                        model.fc.weight.data[true_label] = (
                            momentum * model.fc.weight.data[true_label] +
                            (1-momentum) * new_proto
                        )
                        updated += 1
    
    # ========== 特征可视化6: 原型更新效果 ==========
    if updated > 0:
        with torch.no_grad():
            # 获取原型和新特征
            old_protos = model.fc.weight.data.clone()
            new_protos = torch.stack([model.encode(data[i]) for i in range(len(data))])
            
            # 调整维度为2D
            proto_2d = old_protos.cpu().numpy().reshape(-1, 16)  # 512=16x32
            new_feat_2d = new_protos.mean(0).unsqueeze(0).cpu().numpy().reshape(-1, 16)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            sns.heatmap(proto_2d, cmap='viridis')
            plt.title("Prototypes Before Update")
            
            plt.subplot(122)
            sns.heatmap(new_feat_2d, cmap='viridis')
            plt.title("Average New Features (Reshaped)")
            
            plt.savefig(os.path.join(vis_dir, "6_prototype_update.png"))
            plt.close()
    
    print(f"Session {session}: acc_unknown={acc:.4f}, updated={updated} "
          f"(alpha={alpha:.2f}, momentum={momentum:.2f})")
    return acc
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
