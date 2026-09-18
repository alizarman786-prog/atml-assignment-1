"""Extract frozen backbone features for a whole dataset split.

Kept separate from training so features can be cached and reused across the
head-training step and every later intervention (color, translation, patch
shuffle) without re-running the backbone forward pass redundantly within one
script invocation. (Cross-invocation caching to disk can be added later if
extraction time becomes a bottleneck; not required for correctness.)
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def _collate_with_preprocess(batch, preprocess):
    imgs, labels = zip(*batch)
    tensors = torch.stack([preprocess(img) for img in imgs])
    labels = torch.tensor(labels, dtype=torch.long)
    return tensors, labels


@torch.no_grad()
def extract_features(backbone, dataset, batch_size: int = 64, device: str = "cpu",
                      num_workers: int = 2):
    """Runs `dataset` through `backbone.preprocess` then `backbone.extract_features`.
    Returns (features: FloatTensor[N, D], labels: LongTensor[N])."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_with_preprocess(batch, backbone.preprocess),
    )
    backbone = backbone.to(device).eval()

    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone.extract_features(imgs)
        all_feats.append(feats.cpu())
        all_labels.append(labels)

    return torch.cat(all_feats), torch.cat(all_labels)