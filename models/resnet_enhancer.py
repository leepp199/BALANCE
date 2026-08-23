import torch
import torch.nn as nn  
import torch.nn.functional as F
from sklearn.cluster import KMeans

class FeatureFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, global_feat, local_feat):
        combined = torch.cat([global_feat, local_feat], dim=1)
        alpha = self.mlp(combined)
        return alpha * global_feat + (1-alpha) * local_feat

class EnhancedPositionEncoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.time_enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,1), padding=(1,0)),
            nn.GELU(),
            nn.Conv2d(16, feat_dim, kernel_size=(3,1), padding=(1,0))
        )
        self.freq_enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,3), padding=(0,1)),
            nn.GELU(),
            nn.Conv2d(16, feat_dim, kernel_size=(1,3), padding=(0,1))
        )
        self.gate = nn.Sequential(
            nn.Conv2d(feat_dim*2, feat_dim//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim//2, feat_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, grid):
        assert grid.size(1) == 2, f"输入需要2个坐标通道，实际{grid.size(1)}"
        time_emb = self.time_enc(grid[:,:1])
        freq_emb = self.freq_enc(grid[:,1:])
        combined = torch.cat([time_emb, freq_emb], dim=1)
        gate = self.gate(combined)
        return gate * time_emb + (1-gate) * freq_emb


class LocalFeatureCluster(nn.Module):
    def __init__(self, feat_dim=256, k_ratio=0.2, temporal_scale=1.0):
        super().__init__()
        self.feat_dim = feat_dim  # 与插入层的通道数匹配（如layer3的256）
        self.k_ratio = k_ratio    # 聚类比例调小（中间层特征尺寸大，避免过聚类）
        self.temporal_scale = temporal_scale
        self._current_device = None
        
        # 位置编码器（通道数与feat_dim一致）
        self.pos_encoder = EnhancedPositionEncoder(feat_dim)
        
        # 空间权重网络（适配当前通道数）
        self.spatial_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, 1),
            nn.Sigmoid()
        )
        
        # 残差连接（确保特征增强不破坏原特征）
        self.shortcut = nn.Identity()  # 跳跃连接，原特征直接传递

    def forward(self, features):
        # 输入形状：[B, C, H, W]，输出保持相同形状（供后续层使用）
        B, C, H, W = features.shape
        device = features.device
        self._ensure_device_consistency(device)
        residual = self.shortcut(features)  # 原特征保留（残差）
        
        # 1. 位置编码（增强空间感知）
        grid = self._generate_grid(H, W, device)  # [1,2,H,W]
        pos_emb = self.pos_encoder(grid)  # [1,C,H,W]
        enhanced_feat = features + pos_emb  # 特征+位置编码（残差增强）
        
        # 2. 展平特征用于聚类（[B, H*W, C]）
        flat_feat = enhanced_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        HW = H * W
        
        # 3. 动态聚类（优化：批量处理，消除样本循环）
        k = max(2, int(HW * self.k_ratio))  # 聚类数随特征尺寸动态调整
        
        # 3.1 构建空间位置相似度矩阵（基于坐标距离）
        positions = torch.stack(torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        ), dim=-1).float().view(-1, 2)  # [HW, 2]
        spatial_sim = torch.exp(-torch.cdist(positions, positions) / self.temporal_scale)  # [HW, HW]
        
        # 3.2 批量KMeans聚类（一次性处理整个batch，减少CPU交互）
        with torch.no_grad():
            # 特征重塑为[B*HW, C]，一次性迁移到CPU
            flat_np = flat_feat.detach().cpu().numpy().reshape(-1, C)
            # 批量聚类所有样本（仅一次CPU-GPU交互）
            kmeans = KMeans(n_clusters=k, n_init=10).fit(flat_np)
            # 标签重塑为[B, HW]并迁移回GPU
            labels = torch.from_numpy(kmeans.labels_).to(device).view(B, HW)
            # 聚类中心迁移回GPU
            centers = torch.from_numpy(kmeans.cluster_centers_).float().to(device)  # [k, C]
        
        # 3.3 基于空间相似度加权聚类中心（修正维度匹配问题）
        clustered_feat_list = []
        for b in range(B):
            # 当前样本的特征和标签
            feat_b = flat_feat[b]  # [HW, C]
            label_b = labels[b]    # [HW]
            
            # 生成所有类别的掩码 [k, HW]（2维张量）
            cluster_masks = torch.stack([(label_b == c) for c in range(k)], dim=0)
            # 过滤空聚类
            valid_clusters = cluster_masks.sum(dim=1) > 0
            
            # 向量化计算加权中心（修正维度和einsum方程）
            weighted_centers = torch.zeros(k, C, device=device)
            if valid_clusters.any():
                # 有效聚类的掩码和特征
                valid_masks = cluster_masks[valid_clusters]  # [k_valid, HW]（2维）
                valid_centers = centers[valid_clusters]      # [k_valid, C]
                
                # 计算每个有效聚类的权重（空间相似度平均）
                # 修正1：einsum方程适配2维输入，扩展为3维计算后压缩
                weights = torch.einsum('kh, hw->khw', valid_masks, spatial_sim)  # [k_valid, HW, HW]
                weights = weights.mean(dim=2)  # 沿空间维度平均，得到[K_valid, HW]
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)     # 归一化
                
                # 加权求和计算新中心
                # 修正2：适配2维权重和特征的矩阵乘法
                weighted_feat = torch.einsum('kh, hw->kw', weights, feat_b)    # [k_valid, C]
                weighted_centers[valid_clusters] = weighted_feat
            
            # 空聚类使用原始中心
            weighted_centers[~valid_clusters] = centers[~valid_clusters]
            
            # 聚类结果还原为空间形状 [C, H, W]
            clustered_flat = weighted_centers[label_b]  # [HW, C]
            clustered_feat = clustered_flat.permute(1, 0).view(C, H, W)  # [C, H, W]
            clustered_feat_list.append(clustered_feat)
        
        # 4. 聚类特征与原特征融合（空间权重控制融合比例）
        clustered_feat = torch.stack(clustered_feat_list)  # [B, C, H, W]
        # 计算空间权重（基于原特征的重要性）
        flat_residual = residual.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        spatial_weight = self.spatial_net(flat_residual).permute(0, 2, 1).view(B, 1, H, W)  # [B,1,H,W]
        
        # 融合：聚类特征（局部细节）* 权重 + 原增强特征（全局结构）* (1-权重)
        fused_feat = spatial_weight * clustered_feat + (1 - spatial_weight) * enhanced_feat
        
        # 5. 最终输出：融合特征 + 残差（确保与输入形状一致）
        output = fused_feat + residual  # 残差连接，稳定训练
        return output  # [B, C, H, W]，可直接输入下一层

    def _generate_grid(self, H, W, device):
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1,2,H,W]

    def _ensure_device_consistency(self, device):
        if self._current_device != device:
            self.to(device)
            self._current_device = device
    