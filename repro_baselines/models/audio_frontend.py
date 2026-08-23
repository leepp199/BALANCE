"""Mel-spectrogram frontend for raw audio waveforms.

Transforms raw audio to 3-channel mel-spectrogram images suitable
for ImageNet-pretrained ResNet18 (first conv expects 3 channels).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class AudioFrontend(nn.Module):
    """Convert raw audio waveform → 3-channel mel-spectrogram.

    Uses the same LS-100 / NSynth / FSC-89 preprocessing as PITEL-CUSC's
    official code (``get_mel`` pattern): spectrogram → logmel → bn0 →
    replicate to 3 channels.
    """

    def __init__(self, sample_rate: int = 16000,
                 window_size: int = 400,
                 hop_size: int = 160,
                 mel_bins: int = 128,
                 fmin: int = 0,
                 fmax: int = 8000,
                 window: str = 'hann'):
        super().__init__()
        center = True
        pad_mode = 'reflect'

        self.spectrogram = Spectrogram(
            n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window,
            center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        self.logmel = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax,
            ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True)

        # BN on mel bins (same as PITEL-CUSC / MYNET)
        self.bn0 = nn.BatchNorm2d(mel_bins)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T_raw) → mel: (B, 3, T_mel, mel_bins)"""
        # (B, 1, T, freq)
        x = self.spectrogram(waveform)
        # (B, 1, T, mel_bins)
        x = self.logmel(x)
        # BN on mel bins: (B, mel_bins, T, 1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # Repeat to 3 channels for ImageNet-pretrained conv1
        x = x.repeat(1, 3, 1, 1)
        return x
