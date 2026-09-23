"""Task 4, Step 2: MLS post-hoc unknownness score.

Operates directly on saved logits (N, num_classes numpy array). Larger
values = more "unknown-looking".
"""
from __future__ import annotations

import numpy as np


def mls_score(logits: np.ndarray) -> np.ndarray:
    """u_MLS(x) = -max_k z_k(x). Retains absolute logit magnitude (unlike
    MSP, which normalizes it away via softmax)."""
    return -logits.max(axis=1)
