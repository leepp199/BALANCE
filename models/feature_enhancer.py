import torch.nn as nn
import torch
import math
import torch.nn.functional as F
class TemporalConstraint(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=feat_dim//2,
            bidirectional=True,
            batch_first=True
        )
        # 修改注意力层结构
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.Tanh(),
            nn.Linear(feat_dim//2, 1, bias=False)  # 移除偏置确保数值稳定性
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B,T,feat_dim]
        # 确保注意力输出是浮点张量
        attn_weights = F.softmax(self.attention(lstm_out).float(), dim=1)
        return torch.sum(lstm_out * attn_weights, dim=1)  # [B,feat_dim]

class EnhancedLocalFeature(nn.Module):
    def __init__(self, feat_dim=512, k_ratio=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_centroids = int(feat_dim * k_ratio)
        
        # 改进的时序网络结构
        self.temporal_net = nn.Sequential(
            nn.LSTM(feat_dim, feat_dim//2,  # 严格保持输入维度
                   bidirectional=True,
                   batch_first=True),
            nn.Linear(feat_dim, feat_dim)
        )
        
        # 维度修正层
        self.dim_adjuster = nn.Linear(532, 512) if feat_dim == 512 else nn.Identity()
        
        # 原型中心初始化
        self.centroids = nn.Parameter(
            torch.randn(self.num_centroids, feat_dim) * 0.01
        )

    def forward(self, features):
        B, C, H, W = features.shape
        # 1. 维度修正
        if C != self.feat_dim:
            features = self.dim_adjuster(features.permute(0,2,3,1)).permute(0,3,1,2)
            C = features.size(1)
        
        HW = H * W
        spatial_feat = features.view(B, C, HW)
        
        # 2. 时序特征提取（添加维度验证）
        temporal_input = spatial_feat.permute(0, 2, 1)
        assert temporal_input.size(-1) == self.feat_dim, \
            f"LSTM输入维度应为{self.feat_dim}，实际得到{temporal_input.size(-1)}"
            
        temporal_out, _ = self.temporal_net[0](temporal_input)
        temporal_feat = self.temporal_net[1](temporal_out.mean(1))
        
        # 3. 动态权重计算
        weights = torch.matmul(
            temporal_input,  # [B,HW,C]
            self.centroids.t()  # [C,k]
        ) / math.sqrt(C)
        weights = F.softmax(weights.clamp(min=-50, max=50), dim=-1)
        
        # 4. 特征聚合
        enhanced = torch.bmm(
            spatial_feat,  # [B,C,HW]
            weights  # [B,HW,k]
        ).permute(0, 2, 1)  # [B,k,C]
        
        # 5. 安全reshape
        try:
            scale_factor = C // HW
            if scale_factor * HW != C:
                # 自动填充方案
                pad_size = HW - (C % HW)
                enhanced = F.pad(enhanced, (0, pad_size))
                scale_factor += 1
                
            return enhanced.view(B, self.num_centroids, H, W, scale_factor)\
                         .sum(dim=-1)
        except Exception as e:
            print(f"调试信息：\n输入形状：{features.shape}\n增强特征：{enhanced.shape}\n目标形状：[{B},{self.num_centroids},{H},{W}]")
            raise ValueError(f"维度转换失败：{str(e)}") from e