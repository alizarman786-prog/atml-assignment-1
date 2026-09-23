"""Task 4, Step 4: manifold mixup utility for PROSER's data placeholders.

Mixes two examples' intermediate features (after layer2, before layer3;
see models/resnet_cifar.py's forward_features_upto_layer2/forward_from_layer3
split point) with a Beta(2,2)-sampled coefficient, using only pairs from
DIFFERENT known classes (mixing same-class examples wouldn't produce a
meaningful "between classes" proxy unknown).
"""
from __future__ import annotations

import torch
from torch.distributions import Beta


def make_different_class_pairs(labels: torch.Tensor) -> torch.Tensor:
    """For each index i in the batch, finds a partner index j (possibly
    reusing partners) such that labels[j] != labels[i]. Returns a LongTensor
    of partner indices, same length as labels. If a batch is degenerate
    (all one class), falls back to a random index (no different-class pair
    exists -- documented edge case, extremely unlikely at batch=128 with 10
    classes)."""
    n = labels.size(0)
    device = labels.device
    partner = torch.empty(n, dtype=torch.long, device=device)
    labels_list = labels.tolist()

    for i in range(n):
        candidates = [j for j in range(n) if labels_list[j] != labels_list[i]]
        if not candidates:
            partner[i] = torch.randint(0, n, (1,), device=device).item()
        else:
            partner[i] = candidates[torch.randint(0, len(candidates), (1,)).item()]
    return partner


def manifold_mixup(feat_a: torch.Tensor, feat_b: torch.Tensor, alpha: float = 2.0):
    """Mixes two batches of intermediate features with lambda ~ Beta(alpha, alpha).
    Returns (mixed_feat, lam) -- lam is a scalar float (one shared mixing
    coefficient per call, matching the assignment's single-lambda-per-batch
    formulation: h~ = lambda*h_i + (1-lambda)*h_j)."""
    lam = Beta(alpha, alpha).sample().item()
    mixed = lam * feat_a + (1 - lam) * feat_b
    return mixed, lam
