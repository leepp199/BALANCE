"""Standard audio encoder (NOT MYNET) for CIL baseline methods.

Provides a clean ResNet18 (torchvision, ImageNet-pretrained) with
mel-spectrogram frontend. No extra attention, no hgnn, no feature
enhancers — this is the standard architecture used by CEC, PAN, etc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .audio_frontend import AudioFrontend


class AudioResNet(nn.Module):
    """Standard ResNet18 audio encoder.

    Architecture:
      raw audio → AudioFrontend → ResNet18 (torchvision, pretrained) → fc

    The encoder outputs 512-d features via adaptive average pooling.
    Subclasses can override ``build_classifier`` to add method-specific heads.
    """

    def __init__(self, num_classes: int, pretrained: bool = True,
                 num_features: int = 512):
        super().__init__()
        self.num_features = num_features

        # Audio frontend (spectrogram → logmel → bn0 → repeat 3ch)
        self.frontend = AudioFrontend()

        # Standard torchvision ResNet18 (ImageNet pretrained)
        rn = torchvision.models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None)

        # Remove the original fc (we add our own)
        self.features = nn.Sequential(*list(rn.children())[:-2])
        # After removing avgpool and fc, we add our own
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.fc = nn.Linear(num_features, num_classes, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-d feature vectors from raw audio.

        Args:
            x: (B, T) raw audio waveform.

        Returns:
            (B, D) feature vectors.
        """
        # Audio → mel-spectrogram (3-channel)
        x = self.frontend(x)

        # ResNet18 feature extractor
        x = self.features(x)       # (B, 512, H, W) — actually 256 in our case
        x = self.avgpool(x)        # (B, 512, 1, 1)
        x = torch.flatten(x, 1)    # (B, 512)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and classify.

        Args:
            x: (B, T) raw audio.

        Returns:
            (B, C) logits.
        """
        f = self.encode(x)
        return F.linear(F.normalize(f, dim=-1),
                        F.normalize(self.fc.weight, dim=-1))

    def get_logits(self, feats: torch.Tensor,
                   protos: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
        """Cosine-similarity logits."""
        return temperature * F.linear(
            F.normalize(feats, dim=-1),
            F.normalize(protos, dim=-1))
