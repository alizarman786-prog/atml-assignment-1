"""Task 4, Step 2: Mahalanobis post-hoc unknownness score.

Uses the 512-d penultimate FEATURE (not logits). Class means mu_c and a
single shared DIAGONAL covariance Sigma are estimated from unaugmented
CIFAR-10 training features (per the assignment), with 1e-6 added to every
diagonal entry for numerical stability. Larger score = more "unknown-looking".
"""
from __future__ import annotations

import numpy as np


def fit_mahalanobis(train_features: np.ndarray, train_labels: np.ndarray,
                     num_classes: int, eps: float = 1e-6):
    """Returns (means, diag_var): means is (num_classes, feature_dim),
    diag_var is (feature_dim,) -- the shared diagonal covariance."""
    feature_dim = train_features.shape[1]
    means = np.zeros((num_classes, feature_dim), dtype=np.float64)
    for c in range(num_classes):
        means[c] = train_features[train_labels == c].mean(axis=0)

    centered = train_features - means[train_labels]
    diag_var = centered.var(axis=0) + eps
    return means, diag_var


def mahalanobis_score(features: np.ndarray, means: np.ndarray, diag_var: np.ndarray) -> np.ndarray:
    """u_Mah(x) = min_c (f(x)-mu_c)^T Sigma^-1 (f(x)-mu_c), with Sigma
    diagonal, so this reduces to a per-dimension weighted squared distance
    summed over dimensions, minimized over known classes c."""
    diffs = features[:, None, :] - means[None, :, :]          # (N, C, D)
    sq_weighted = (diffs ** 2) / diag_var[None, None, :]        # (N, C, D)
    dist_per_class = sq_weighted.sum(axis=2)                    # (N, C)
    return dist_per_class.min(axis=1)
