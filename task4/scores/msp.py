"""Task 4, Step 2: MSP post-hoc unknownness score.

Operates directly on saved logits (N, num_classes numpy array). Larger
values = more "unknown-looking".
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax


def msp_score(logits: np.ndarray) -> np.ndarray:
    """u_MSP(x) = 1 - max_k p_k(x), p = softmax(logits). Normalized
    confidence: ignores absolute logit scale, only the shape of the
    softmax distribution."""
    probs = softmax(logits, axis=1)
    return 1.0 - probs.max(axis=1)
