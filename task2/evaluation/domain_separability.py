"""Task 2 Step 5: domain separability diagnostic.

Freezes the given model's backbone, extracts features for equal numbers of
source-validation and target examples, and trains a balanced logistic
regression classifier (C=1, seed-6304 70/30 split) to distinguish source
from target. Held-out accuracy is the domain separability score; 50%
indicates chance (domains fully confused), 100% indicates domains are
trivially separable.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(model, dataset, device: str, transform, batch_size: int = 64) -> np.ndarray:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: torch.stack([transform(img) for img, _ in b]),
    )
    model.eval()
    feats = []
    for imgs in loader:
        feats.append(model.forward_features(imgs.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def domain_separability(
    model, source_val_datasets: dict, target_dataset, device: str, transform,
    seed: int = 6304, C: float = 1.0,
) -> dict:
    """source_val_datasets: {domain: Dataset} (the three source validation
    splits, pooled). target_dataset: full target-domain Dataset."""
    rng = np.random.default_rng(seed)

    source_feats_list = [
        extract_features(model, ds, device, transform) for ds in source_val_datasets.values()
    ]
    source_feats = np.concatenate(source_feats_list, axis=0)
    target_feats = extract_features(model, target_dataset, device, transform)

    n = min(len(source_feats), len(target_feats))
    src_idx = rng.choice(len(source_feats), size=n, replace=False)
    tgt_idx = rng.choice(len(target_feats), size=n, replace=False)
    X = np.concatenate([source_feats[src_idx], target_feats[tgt_idx]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y,
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)
    held_out_accuracy = clf.score(X_test, y_test)

    return {
        "n_per_class": n,
        "domain_separability_accuracy": float(held_out_accuracy),
    }
