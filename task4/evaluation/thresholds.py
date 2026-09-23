"""Task 4, Step 6: validation-calibrated rejection threshold.

Threshold = 95th percentile of unknownness on the CIFAR-10 VALIDATION set
(known data only) -- accept x when u(x) <= tau, aiming to accept ~95% of
known examples. Achieved test-set acceptance rate and near/far rejection
rates (equivalently FPR@95TPR: the fraction of unknown examples incorrectly
accepted) are then reported using this fixed threshold.
"""
from __future__ import annotations

import numpy as np


def calibrate_threshold(val_scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(val_scores, percentile))


def calibrated_rejection_summary(
    threshold: float, test_known_scores: np.ndarray,
    near_scores: np.ndarray, far_scores: np.ndarray,
) -> dict:
    """accept x when u(x) <= threshold."""
    known_acceptance_rate = float((test_known_scores <= threshold).mean())
    near_acceptance_rate = float((near_scores <= threshold).mean())  # "incorrectly accepted"
    far_acceptance_rate = float((far_scores <= threshold).mean())
    all_unknown_scores = np.concatenate([near_scores, far_scores])
    all_acceptance_rate = float((all_unknown_scores <= threshold).mean())

    return {
        "threshold": threshold,
        "known_test_acceptance_rate": known_acceptance_rate,
        "near_unknown_rejection_rate": 1.0 - near_acceptance_rate,
        "far_unknown_rejection_rate": 1.0 - far_acceptance_rate,
        "near_fpr_at_95tpr": near_acceptance_rate,
        "far_fpr_at_95tpr": far_acceptance_rate,
        "all_unknown_rejection_rate": 1.0 - all_acceptance_rate,
        "all_fpr_at_95tpr": all_acceptance_rate,
    }
