"""Representation stability (I_T) for Task 1 Step 6."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.metrics import cosine_similarity_pairs  # noqa: E402


def representation_stability(clean_feats: np.ndarray, transformed_feats: np.ndarray) -> float:
    """Mean paired cosine similarity I_T = (1/N) sum_i cos(f(x_i), f(T(x_i))),
    the assignment's exact formula. clean_feats[i] and transformed_feats[i]
    must correspond to the same underlying example."""
    sims = cosine_similarity_pairs(clean_feats, transformed_feats)
    return float(np.mean(sims))
