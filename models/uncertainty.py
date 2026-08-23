import torch
import numpy as np
from tqdm import tqdm

def get_base_class_uncertainty(model, base_loader, device, k_dropout=5, a_mask=4):
    """
    计算所有基础类的不确定度分数。
    修正方案：直接调用 model.get_uncertainty 方法。
    该方法内部会自动处理：Raw Audio -> Spectrogram -> Mask Augmentation -> MC Dropout
    """
    model.eval()
    
    class_unc_sum = {}
    class_counts = {}
    
    print(f"==> [UNCG] Calculating Uncertainty using model built-in method (K={k_dropout}, A={a_mask})...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(base_loader, desc="Uncertainty Est")):
            # 获取数据 (根据你的dataloader，通常是 data, label)
            data, labels = batch[0].to(device), batch[1].to(device)
            
            # 关键修改：直接调用模型内部的 get_uncertainty
            # 它接受原始音频 (B, T)，内部转为声谱图并做 mask
            if hasattr(model, 'module'):
                # 如果是 DataParallel 包装的模型
                batch_uncs = model.module.get_uncertainty(data, n_aug=a_mask, n_forward=k_dropout)
            else:
                batch_uncs = model.get_uncertainty(data, n_aug=a_mask, n_forward=k_dropout)
            
            # 转换结果为 numpy
            if isinstance(batch_uncs, torch.Tensor):
                batch_uncs = batch_uncs.cpu().numpy()
            elif isinstance(batch_uncs, float): # batch size=1 的情况
                batch_uncs = [batch_uncs]
                
            labels = labels.cpu().numpy()
            
            # 累加每个类的不确定度
            for lbl, unc in zip(labels, batch_uncs):
                if lbl not in class_unc_sum:
                    class_unc_sum[lbl] = 0.0
                    class_counts[lbl] = 0
                class_unc_sum[lbl] += unc
                class_counts[lbl] += 1

    # 计算平均值
    final_scores = {}
    for cls in class_unc_sum:
        if class_counts[cls] > 0:
            final_scores[cls] = class_unc_sum[cls] / class_counts[cls]
        else:
            final_scores[cls] = 0.0 # 默认
            
    return final_scores
