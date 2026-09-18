"""Frozen backbone wrappers for Task 1.

Each wrapper exposes:
  - `preprocess`: the torchvision transform (resize/crop/normalize) required
    by that specific pretrained model, applied AFTER a shared 224x224 RGB
    intervention has already been constructed (per the assignment: build
    interventions on a common 224x224 image, then apply each model's own
    normalization).
  - `extract_features(images) -> Tensor[N, D]`: the frozen final
    representation (GAP feature for ResNet, CLS token for ViT, normalized
    embedding for CLIP).
  - `zero_shot_predict(images, class_names)`: CLIP-only, using the fixed
    prompt template required by the assignment.

Backbones are frozen (`requires_grad_(False)`, `eval()`); only the separate
`LinearHead` module is trained.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

PROMPT_TEMPLATE = "a photo of a {}."


class LinearHead(nn.Module):
    """Trained on top of a frozen backbone's extracted features."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats)


@dataclass
class BackboneOutput:
    features: torch.Tensor  # (N, D), frozen representation
    feature_dim: int


class ResNet50Backbone(nn.Module):
    """torchvision ResNet-50, ImageNet1K_V2 weights. Feature = global-average
    -pooled activation before the final fc (2048-d)."""

    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        net = resnet50(weights=weights)
        self.feature_dim = net.fc.in_features  # 2048
        net.fc = nn.Identity()
        self.net = net.eval()
        self.net.requires_grad_(False)
        # Use the weights' own preprocessing (resize/crop/normalize) so it
        # matches how the model was trained; assignment says to build
        # interventions on a common 224x224 image first, then apply this.
        self.preprocess = weights.transforms()

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class ViTB16Backbone(nn.Module):
    """torchvision ViT-B/16, ImageNet1K_V1 weights. Feature = final class
    token (768-d), taken before the classification head."""

    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        net = vit_b_16(weights=weights)
        self.feature_dim = net.hidden_dim  # 768
        net.heads = nn.Identity()
        self.net = net.eval()
        self.net.requires_grad_(False)
        self.preprocess = weights.transforms()

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        # torchvision's ViT forward with heads=Identity already returns the
        # CLS token representation.
        return self.net(images)


class CLIPBackbone(nn.Module):
    """OpenCLIP ViT-B-32, pretrained='openai'. Feature = L2-normalized image
    embedding (512-d). Also supports zero-shot classification with the
    assignment's fixed prompt template."""

    def __init__(self, class_names: list[str] | None = None):
        super().__init__()
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.preprocess = preprocess
        self.feature_dim = model.visual.output_dim  # 512
        self.logit_scale = model.logit_scale.exp().item()

        self._text_features = None
        if class_names is not None:
            self.set_classes(class_names)

    @torch.no_grad()
    def set_classes(self, class_names: list[str]):
        """Precompute normalized text embeddings for zero-shot prediction
        using the required fixed prompt template."""
        prompts = [PROMPT_TEMPLATE.format(c) for c in class_names]
        tokens = self.tokenizer(prompts)
        text_feats = self.model.encode_text(tokens)
        self._text_features = F.normalize(text_feats, dim=-1)
        self.class_names = class_names

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def zero_shot_predict(self, images: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities over class_names using scaled
        cosine similarity, as required for the zero-shot confidence metric."""
        assert self._text_features is not None, "call set_classes() first"
        img_feats = self.extract_features(images)
        logits = self.logit_scale * img_feats @ self._text_features.T
        return F.softmax(logits, dim=-1)


def build_backbone(name: str, class_names: list[str] | None = None) -> nn.Module:
    """Factory. name in {'resnet50', 'vit_b_16', 'clip_vit_b_32'}."""
    if name == "resnet50":
        return ResNet50Backbone()
    if name == "vit_b_16":
        return ViTB16Backbone()
    if name == "clip_vit_b_32":
        return CLIPBackbone(class_names=class_names)
    raise ValueError(f"Unknown backbone {name!r}")
