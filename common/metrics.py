"""Metrics shared across tasks: top-1 accuracy, macro-F1, prediction consistency.

Kept dependency-light (numpy + sklearn) so any task can import without pulling
in task-specific code.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def top1_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(preds == labels))


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, preds, average="macro"))


def prediction_consistency(preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    """Fraction of examples whose predicted class is unchanged between two
    prediction arrays (e.g. clean vs. transformed). Used in Task 1 (color,
    translation, patch-shuffle) and generally useful anywhere two prediction
    sets over the same examples need to be compared."""
    assert preds_a.shape == preds_b.shape
    return float(np.mean(preds_a == preds_b))


def mean_max_confidence(probs: np.ndarray) -> float:
    """Mean of the max-class softmax probability per example."""
    return float(np.mean(np.max(probs, axis=1)))


def cosine_similarity_pairs(feat_a: np.ndarray, feat_b: np.ndarray) -> np.ndarray:
    """Per-example cosine similarity between two (N, D) feature arrays.
    Used for the Task 1 representation-stability index I_T."""
    a_norm = feat_a / np.linalg.norm(feat_a, axis=1, keepdims=True)
    b_norm = feat_b / np.linalg.norm(feat_b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)
