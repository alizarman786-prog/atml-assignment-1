"""ResNet-18 backbone for Task 2 (and Task 3, which reuses this unchanged).

Implements the assignment's exact BatchNorm policy: freeze BatchNorm
RUNNING STATISTICS (mean/var) at their pretrained ImageNet values for every
method, so the source/target mixture in adaptation batches cannot silently
smuggle in an implicit form of adaptation via updated BN statistics. The
affine parameters (gamma/beta) remain trainable. Per the assignment: after
calling model.train(), place only the BatchNorm modules in eval mode
(freeze_batchnorm_running_stats below) -- never call model.eval() on the
whole network during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class PACSBackbone(nn.Module):
    """torchvision ResNet-18 (ImageNet1K_V1 weights) with its final fc
    replaced by a 512 -> num_classes linear head. The whole network
    (backbone + head) is fine-tuned; only BatchNorm running stats are
    frozen (see freeze_batchnorm_running_stats)."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        self.feature_dim = net.fc.in_features  # 512
        net.fc = nn.Linear(self.feature_dim, num_classes)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits."""
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 512-d penultimate feature (before fc), used by every
        adaptation method (DAN/DANN/CDAN/DAN-DG operate on this feature)."""
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
        """Applies just the final linear head to an already-computed
        feature (used when a method needs the feature AND the logits from
        one forward pass, e.g. CDAN's classifier probabilities)."""
        return self.net.fc(feat)


def freeze_batchnorm_running_stats(model: nn.Module) -> None:
    """Call this every time after model.train() (never call model.eval() on
    the whole network during training). Puts only BatchNorm2d modules into
    eval mode so their running_mean/running_var stop updating, while every
    other module (including BatchNorm's own affine weight/bias, which stay
    trainable) remains in train mode."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def build_pacs_model(num_classes: int = 7) -> PACSBackbone:
    return PACSBackbone(num_classes=num_classes)
