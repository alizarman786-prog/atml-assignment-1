"""Task 2 Step 5: per-class target accuracy and comparison against the
Source-only baseline, to surface class-specific negative transfer that an
aggregate accuracy/macro-F1 number can hide.
"""
from __future__ import annotations

import numpy as np


def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            acc[c] = (preds[mask] == labels[mask]).mean()
        else:
            acc[c] = float("nan")
    return acc


def compare_to_baseline(
    baseline_preds: np.ndarray, method_preds: np.ndarray, labels: np.ndarray,
    class_names: list[str], top_k: int = 3,
) -> dict:
    """Returns per-class accuracy for both, the delta, and the top_k most
    improved / most degraded classes (method vs. baseline)."""
    num_classes = len(class_names)
    baseline_acc = per_class_accuracy(baseline_preds, labels, num_classes)
    method_acc = per_class_accuracy(method_preds, labels, num_classes)
    delta = method_acc - baseline_acc

    order = np.argsort(delta)
    most_degraded = [
        {"class": class_names[i], "baseline_acc": float(baseline_acc[i]),
         "method_acc": float(method_acc[i]), "delta": float(delta[i])}
        for i in order[:top_k]
    ]
    most_improved = [
        {"class": class_names[i], "baseline_acc": float(baseline_acc[i]),
         "method_acc": float(method_acc[i]), "delta": float(delta[i])}
        for i in order[::-1][:top_k]
    ]

    return {
        "per_class_baseline_acc": {class_names[i]: float(baseline_acc[i]) for i in range(num_classes)},
        "per_class_method_acc": {class_names[i]: float(method_acc[i]) for i in range(num_classes)},
        "most_improved": most_improved,
        "most_degraded": most_degraded,
    }
