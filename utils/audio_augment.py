import torchaudio
import random
import torch
class AudioAugment:
    def __init__(self, sample_rate=16000, device='cuda'):
        self.sample_rate = sample_rate
        self.device = device
        
    def __call__(self, x):
        """保持批处理维度的增强入口"""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,T] → [B,1,T]
        x = x.to(self.device)
        
        # 随机增强链
        if random.random() > 0.5:
            x = self.time_stretch(x)
        if random.random() > 0.5:
            x = self.pitch_shift(x)
        return x
    def time_stretch(self, x):
        rate = random.uniform(0.8, 1.2)
        return torchaudio.transforms.TimeStretch(hop_length=512)(x, rate)
    
    def pitch_shift(self, x):
        n_steps = random.randint(-2, 2)
        return torchaudio.functional.pitch_shift(x, self.sample_rate, n_steps)
    
    def add_noise(self, x):
        noise = torch.randn_like(x) * 0.01
        return x + noise