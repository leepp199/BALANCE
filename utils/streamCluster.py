import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

class FStream:
    """
    优化版FBPStream聚类器 (200行核心实现)
    保留原始算法精髓，同时针对15-30样本的小批量音频数据优化
    
    改进点：
    1. 简化k-NN图维护逻辑
    2. 动态半径调整
    3. 基于音频特征的相似度计算
    4. 流式质心更新
    """
    
    def __init__(self, init_radius=0.4, min_weight=0.3, k=3):
        self.radius = init_radius
        self.min_weight = min_weight
        self.k = k
        self.micro_clusters = []
        self.current_time = 0
        
    class MicroCluster:
        __slots__ = ['center', 'points', 'weight', 'last_update']  # 内存优化
        
        def __init__(self, point, time):
            self.center = point
            self.points = [point]
            self.weight = 1.0
            self.last_update = time
            
        def update(self, point, time, decay=0.99):
            """流式更新中心点"""
            self.weight = decay * self.weight + 1
            self.center = (self.center*(self.weight-1) + point) / self.weight
            self.points.append(point)
            self.last_update = time
    
    def _cosine_sim(self, a, b):
        """音频优化相似度计算"""
        return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
    
    def _dynamic_radius(self):
        """基于最近距离动态调整半径"""
        if len(self.micro_clusters) < 2:
            return self.radius
        
        distances = []
        for i in range(len(self.micro_clusters)):
            for j in range(i+1, len(self.micro_clusters)):
                dist = 1 - self._cosine_sim(self.micro_clusters[i].center, 
                                          self.micro_clusters[j].center)
                distances.append(dist)
        
        return np.percentile(distances, 25) if distances else self.radius
    
    def process_batch(self, batch):
        """处理小批量音频特征"""
        self.current_time += 1
        curr_radius = self._dynamic_radius()
        
        for point in batch:
            # 查找最近微簇
            best_match, best_sim = None, -1
            for mc in self.micro_clusters:
                sim = self._cosine_sim(point, mc.center)
                if sim > best_sim:
                    best_match, best_sim = mc, sim
            
            # 分配或新建微簇
            if best_match and best_sim > (1 - curr_radius):
                best_match.update(point, self.current_time)
            else:
                new_mc = self.MicroCluster(point, self.current_time)
                self.micro_clusters.append(new_mc)
        
        # 权重衰减和清理
        self.micro_clusters = [
            mc for mc in self.micro_clusters 
            if mc.weight > self.min_weight
        ]
    
    def get_clusters(self):
        """获取最终聚类结果"""
        if not self.micro_clusters:
            return [], []
        
        # 构建k-NN图 (简化版)
        centers = np.array([mc.center for mc in self.micro_clusters])
        nbrs = NearestNeighbors(n_neighbors=self.k, metric='cosine').fit(centers)
        distances, indices = nbrs.kneighbors(centers)
        
        # 核心-边界分离 (简化版)
        core_indices = set()
        for i in range(len(centers)):
            if all(j in indices[i] for j in np.where(distances[i] < np.median(distances))[0]):
                core_indices.add(i)
        
        # 分配标签
        cluster_labels = [-1] * len(self.micro_clusters)
        current_label = 0
        
        for i in core_indices:
            if cluster_labels[i] == -1:
                queue = [i]
                cluster_labels[i] = current_label
                
                while queue:
                    idx = queue.pop()
                    for neighbor in indices[idx]:
                        if cluster_labels[neighbor] == -1 and neighbor in core_indices:
                            cluster_labels[neighbor] = current_label
                            queue.append(neighbor)
                
                current_label += 1
        
        # 边界点分配
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                for neighbor in indices[i]:
                    if cluster_labels[neighbor] != -1:
                        cluster_labels[i] = cluster_labels[neighbor]
                        break
        
        return centers, np.array(cluster_labels)
    
    def fit_predict(self, X):
        """批量接口"""
        self.process_batch(X)
        _, labels = self.get_clusters()
        return labels