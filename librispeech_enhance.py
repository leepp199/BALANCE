import os
import numpy as np
import random
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torchvision import transforms
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 必须放在所有matplotlib导入前
import matplotlib.pyplot as plt
from IPython.display import Audio, display

class AdvancedAudioAugmentor:
    """增强版音频数据增强器（支持可配置的多视图生成）"""
    def __init__(self, sample_rate=16000, config=None):
        self.sample_rate = sample_rate
        # 提供默认配置
        self.config = {
            'time_mask_param': 15,
            'freq_mask_param': 15,
            'n_fft': 1024,
            'n_mels': 64,
            'spec_aug': {
                'time_drop_width': 8,
                'time_stripes_num': 2,
                'freq_drop_width': 8,
                'freq_stripes_num': 2
            },
            'aug_prob': {
                'noise': 0.8,
                'pitch_shift': 0.7,
                'time_stretch': 0.5
            },
            'use_noise_lib': False  # 默认关闭噪声库
        }
        
        # 如果提供了config，则更新默认配置
        if config:
            self.config.update(config)
        
        # 其余初始化代码保持不变...
        self.spec_transform = Spectrogram(
            n_fft=self.config['n_fft'],
            hop_length=self.config['n_fft'] // 4,
            win_length=self.config['n_fft']
        )
        
        # 噪声库加载（只有当use_noise_lib为True时才会加载）
        self.noise_files = self._load_noise_library() if self.config['use_noise_lib'] else None

    def _load_noise_library(self, noise_dir='path/to/noise/files'):
        """加载环境噪声库"""
        if os.path.exists(noise_dir):
            return [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
        return None

    def _time_augment(self, audio):
        """增强版时域增强流水线"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # 随机选择增强组合
        aug_seq = random.choice([
            [self._random_crop, self._pitch_shift, self._add_noise],
            [self._time_stretch, self._add_noise],
            [self._random_crop, self._polarity_invert]
        ])
        
        for aug in aug_seq:
            if random.random() < self.config['aug_prob'].get(aug.__name__[1:], 0.5):
                audio = aug(audio.squeeze(0)).unsqueeze(0)
                
        return self.time_mask(self.freq_mask(audio))

    def _random_crop(self, audio, crop_range=(0.7, 1.0)):
        """随机裁剪（保留主要内容）"""
        target_len = int(len(audio) * random.uniform(*crop_range))
        start = random.randint(0, len(audio) - target_len)
        return audio[start:start+target_len]

    def _pitch_shift(self, audio):
        """音高变换（音色保留）"""
        n_steps = random.choice([-3, -2, -1, 1, 2, 3])
        return torchaudio.functional.pitch_shift(audio, self.sample_rate, n_steps)

    def _time_stretch(self, audio, rate_range=(0.8, 1.2)):
        """时间拉伸（保持音高）"""
        rate = random.uniform(*rate_range)
        return torchaudio.functional.time_stretch(audio, self.sample_rate, rate)

    def _add_noise(self, audio):
        """智能添加噪声"""
        if self.noise_files and random.random() < 0.6:
            # 使用真实环境噪声
            noise_file = random.choice(self.noise_files)
            noise, _ = torchaudio.load(noise_file)
            noise = noise[:len(audio)] if len(noise) > len(audio) else noise
            snr_db = random.uniform(10, 20)
        else:
            # 使用高斯噪声
            noise = torch.randn_like(audio)
            snr_db = random.uniform(15, 25)
            
        noise = noise * (10 ** (-snr_db / 20))
        return audio + noise

    def _polarity_invert(self, audio):
        """极性反转（相位变化）"""
        return -audio

    def __call__(self, audio):
        """生成3种增强视图（时域+频域+混合）"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # 视图1：纯时域增强
        view1 = self._time_augment(audio.clone())
        
        # 视图2：纯频域增强
        spec = self.spec_transform(audio)
        logmel = self.logmel(spec)
        view2 = self.spec_augment(logmel)
        
        # 视图3：时频混合增强
        temp = self._time_augment(audio.clone())
        spec_temp = self.spec_transform(temp)
        view3 = self.logmel(spec_temp)
        
        return view1.squeeze(0), view2.squeeze(0), view3.squeeze(0)

class EnhancedLBRS(Dataset):
    """增强版音频数据集加载器"""
    def __init__(self, root='/data/datasets/librispeech_fscil/', phase='train', 
                 index=None, k=5, base_sess=None, args=None, session=0):
        self.root = os.path.expanduser(root)
        self.phase = phase
        self.sample_rate = 16000
        self.args = args
        
        # 初始化增强器（训练阶段使用高级增强）
        self.augmenter = AdvancedAudioAugmentor(self.sample_rate) if phase == 'train' else None
        
        # 加载元数据
        self._load_metadata()
        self._select_data(index, k, base_sess, session)

    def _load_metadata(self):
        """加载所有元数据文件"""
        self.meta = {
            'train': pd.read_csv(os.path.join(self.root, "librispeech_fscil_train.csv")),
            'val': pd.read_csv(os.path.join(self.root, "librispeech_fscil_val.csv")),
            'test': pd.read_csv(os.path.join(self.root, "librispeech_fscil_test.csv"))
        }

    def _select_data(self, index, k, base_sess, session):
        """根据阶段选择数据"""
        if self.phase == 'train':
            if base_sess:
                self.data, self.targets = self._select_from_df('train', index)
            else:
                self._handle_few_shot_case(index)
        elif self.phase == 'val':
            self.data, self.targets = self._select_from_df('val', index, per_num=k)
        else:
            self.data, self.targets = self._select_from_df('test', index)

    def _handle_few_shot_case(self, index):
        """处理few-shot学习场景"""
        self.data1, self.targets1 = self._select_from_df('train', index)
        selected_classes = np.random.choice(range(self.args.num_labeled_classes), 5, False)
        self.data2, self.targets2 = self._select_from_df('val', selected_classes, per_num=100)
        self.data = self.data1 + self.data2
        self.targets = self.targets1 + self.targets2

    def _select_from_df(self, partition, index, per_num=None):
        """从DataFrame中选择数据"""
        df = self.meta[partition]
        data, targets = [], []
        
        for class_idx in index:
            class_samples = df[df['label'] == class_idx]
            samples = class_samples[:per_num] if per_num else class_samples
            
            for _, row in samples.iterrows():
                path = os.path.join(self.root, "100spks_segments/", row['filename'])
                if os.path.exists(path):  # 确保文件存在
                    data.append(path)
                    targets.append(row['label'])
                    
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.data[idx])
        audio = audio.squeeze(0)
        
        if self.phase == 'train' and self.augmenter:
            view1, view2, view3 = self.augmenter(audio)
            return {
                'view1': view1,
                'view2': view2,
                'view3': view3,
                'target': self.targets[idx],
                'path': self.data[idx]
            }
        return {
            'audio': audio,
            'target': self.targets[idx],
            'path': self.data[idx]
        }

def visualize_augmentations(sample, augmentor, save_dir='aug_vis'):
    """可视化增强效果对比"""
    os.makedirs(save_dir, exist_ok=True)
    
    original = sample['audio']
    views = augmentor(original.clone())
    
    # 波形对比
    plt.figure(figsize=(15, 9))
    for i, (name, audio) in enumerate(zip(
        ['Original', 'Time-Aug', 'Spec-Aug', 'Mixed-Aug'],
        [original, *views]
    )):
        plt.subplot(4, 2, i*2+1)
        plt.plot(audio.numpy())
        plt.title(f"{name} Waveform")
        
        # 频谱对比
        plt.subplot(4, 2, i*2+2)
        spec = T.Spectrogram(n_fft=1024)(audio.unsqueeze(0)).squeeze(0)
        plt.imshow(10*torch.log10(spec+1e-10), 
                  aspect='auto', 
                  origin='lower',
                  cmap='viridis')
        plt.title(f"{name} Spectrogram")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/aug_comparison.png")
    plt.close()
    
    # 保存音频对比
    for i, (name, audio) in enumerate(zip(
        ['original', 'time_aug', 'spec_aug', 'mixed_aug'],
        [original, *views]
    )):
        torchaudio.save(f"{save_dir}/{name}.wav", audio.unsqueeze(0), sample['sr'])

if __name__ == "__main__":
    # 示例用法
    dataset = EnhancedLBRS(root='/data/datasets/librispeech_fscil/', phase='train')
    sample = dataset[0]
    
    # 初始化增强器
    augmentor = AdvancedAudioAugmentor(sample_rate=16000)
    
    # 可视化增强效果
    visualize_augmentations(sample, augmentor)
    
    # 测试数据加载
    print(f"数据集大小: {len(dataset)}")
    print(f"示例数据形状: {sample['view1'].shape if 'view1' in sample else sample['audio'].shape}")