"""Task 3 Step 4: source-domain separability diagnostic.

Unlike Task 2's binary source-vs-target separability, Task 3 never touches
the target domain during training, so this diagnostic instead asks: can a
simple classifier tell WHICH of the three observed source domains (Photo,
Art Painting, Cartoon) a feature came from? Chance = 33.3%. A lower score
indicates the representation has become more invariant across the domains
it was allowed to see -- it does NOT by itself establish that class
information or unseen-domain (Sketch) performance improved.
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


def source_domain_separability(
    model, source_val_datasets: dict, device: str, transform,
    seed: int = 6304, C: float = 1.0,
) -> dict:
    """source_val_datasets: {domain_name: Dataset} for exactly the three
    source validation splits. Balances to the smallest domain's count,
    70/30 split (seed 6304), multinomial logistic regression, C=1."""
    rng = np.random.default_rng(seed)

    domain_names = sorted(source_val_datasets.keys())
    feats_by_domain = {
        name: extract_features(model, ds, device, transform)
        for name, ds in source_val_datasets.items()
    }
    n = min(len(f) for f in feats_by_domain.values())

    X_parts, y_parts = [], []
    for label_idx, name in enumerate(domain_names):
        feats = feats_by_domain[name]
        idx = rng.choice(len(feats), size=n, replace=False)
        X_parts.append(feats[idx])
        y_parts.append(np.full(n, label_idx))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y,
    )
    clf = LogisticRegression(C=C, multi_class="multinomial", max_iter=1000)
    clf.fit(X_train, y_train)
    held_out_accuracy = clf.score(X_test, y_test)

    return {
        "domain_names": domain_names,
        "n_per_domain": n,
        "chance_level": 1.0 / len(domain_names),
        "source_domain_separability_accuracy": float(held_out_accuracy),
    }
