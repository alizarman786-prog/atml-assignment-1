"""Task 4, Step 6: AUROC computation for Known-vs-Near, Known-vs-Far, and
Known-vs-All-Unknown comparisons, given any unknownness score array.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auroc(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    """AUROC treating 'unknown' as the positive class: y=1 for unknown
    examples, y=0 for known examples, using the unknownness score directly
    (higher = more unknown-looking, matching the positive-class convention)."""
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_score = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(y_true, y_score))


def osr_auroc_summary(known_scores: np.ndarray, near_scores: np.ndarray, far_scores: np.ndarray) -> dict:
    all_unknown_scores = np.concatenate([near_scores, far_scores])
    return {
        "auroc_known_vs_near": compute_auroc(known_scores, near_scores),
        "auroc_known_vs_far": compute_auroc(known_scores, far_scores),
        "auroc_known_vs_all": compute_auroc(known_scores, all_unknown_scores),
    }
