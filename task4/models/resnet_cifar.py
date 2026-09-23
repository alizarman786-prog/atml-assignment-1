"""Task 4: CIFAR-appropriate ResNet-18.

Per the assignment: replace the ImageNet-style 7x7 stride-2 first
convolution with a 3x3 stride-1 convolution, and remove the initial
max-pooling layer -- standard adaptations for 32x32 images, where the
aggressive ImageNet-style early downsampling would throw away too much
spatial resolution immediately. Trained from random initialization (no
pretrained weights) for Vanilla and GCSC; PROSER initializes from the
selected Vanilla checkpoint.

Exposes forward_features_upto_layer2 / forward_from_layer3 as the split
point PROSER's manifold mixup needs ("after layer2 and before layer3").
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net
        self.feature_dim = net.fc.in_features  # 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits."""
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Full penultimate feature (512-d), used for post-hoc scores /
        Mahalanobis distance."""
        n = self.net
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        x = n.layer3(x)
        x = n.layer4(x)
        x = n.avgpool(x)
        return torch.flatten(x, 1)

    def classify_features(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net.fc(feat)

    def forward_features_upto_layer2(self, x: torch.Tensor) -> torch.Tensor:
        """Intermediate feature map after layer2 (before layer3) -- the
        manifold-mixup split point PROSER specifies."""
        n = self.net
        x = n.conv1(x)
        x = n.bn1(x)
        x = n.relu(x)
        x = n.maxpool(x)
        x = n.layer1(x)
        x = n.layer2(x)
        return x

    def forward_from_layer3(self, x: torch.Tensor) -> torch.Tensor:
        """Continues from an already-computed after-layer2 feature map
        through layer3, layer4, and pooling, to the 512-d penultimate
        feature."""
        n = self.net
        x = n.layer3(x)
        x = n.layer4(x)
        x = n.avgpool(x)
        return torch.flatten(x, 1)


def build_cifar_resnet18(num_classes: int = 10) -> CIFARResNet18:
    return CIFARResNet18(num_classes=num_classes)
