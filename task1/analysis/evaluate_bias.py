"""Shape bias and coverage computation for Task 1's shape/texture cue-conflict
analysis (assignment's exact formulas).
"""
from __future__ import annotations

import numpy as np


def classify_predictions(preds: np.ndarray, content_labels: np.ndarray, style_labels: np.ndarray) -> dict:
    """Classifies each prediction as belonging to the content/shape class,
    the style/texture class, or neither ('other'). content_label != style_label
    always holds for cue-conflict images, so these three categories are
    mutually exclusive and exhaustive."""
    is_shape = preds == content_labels
    is_texture = preds == style_labels
    is_other = ~(is_shape | is_texture)
    return {"is_shape": is_shape, "is_texture": is_texture, "is_other": is_other}


def shape_bias_and_coverage(preds: np.ndarray, content_labels: np.ndarray, style_labels: np.ndarray) -> dict:
    """Returns N_shape, N_texture, N_other, N_total, Shape Bias(%), Coverage(%)
    using the assignment's exact formulas:
        Shape Bias(%)  = N_shape / (N_shape + N_texture) * 100
        Coverage(%)    = (N_shape + N_texture) / N_total * 100
    """
    cls = classify_predictions(preds, content_labels, style_labels)
    n_shape = int(cls["is_shape"].sum())
    n_texture = int(cls["is_texture"].sum())
    n_other = int(cls["is_other"].sum())
    n_total = len(preds)

    denom = n_shape + n_texture
    shape_bias_pct = (n_shape / denom * 100) if denom > 0 else float("nan")
    coverage_pct = denom / n_total * 100 if n_total > 0 else float("nan")

    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_other,
        "n_total": n_total,
        "shape_bias_pct": shape_bias_pct,
        "coverage_pct": coverage_pct,
    }
