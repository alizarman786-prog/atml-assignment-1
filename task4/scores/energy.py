"""Task 4, Step 2: Energy post-hoc unknownness score.

Operates directly on saved logits (N, num_classes numpy array). Larger
values = more "unknown-looking".
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def energy_score(logits: np.ndarray) -> np.ndarray:
    """u_Energy(x) = -log sum_k exp(z_k(x)). Uses ALL logits (unlike MLS,
    which only looks at the single largest one)."""
    return -logsumexp(logits, axis=1)
