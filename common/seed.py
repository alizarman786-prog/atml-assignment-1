"""Central seeding utility so every task/script uses the exact same convention.

The assignment repeatedly asks for seed 6304 for splits, subset selection, patch
permutations, and model comparisons. Import `set_seed` everywhere rather than
calling torch/numpy/random seeding ad hoc, so seeding behavior is identical and
auditable across all four tasks.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 6304


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices).

    deterministic=True also asks cuDNN for deterministic algorithms. This can
    slow training down; set False for quick iteration and True for the runs
    you intend to report.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seeded_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """A standalone numpy Generator for subset/split selection, independent of
    global numpy random state (so calling this doesn't perturb other code)."""
    return np.random.default_rng(seed)
