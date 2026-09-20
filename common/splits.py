"""Generic stratified train/val split, shared across tasks. Preserves class
proportions within each split. Used by Task 1's subset selection and Task
2/3's per-domain PACS splits.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from common.seed import seeded_rng


def stratified_split(labels, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """labels: array-like of int class labels, one per example, indexed 0..N-1
    (these indices are relative to whatever pool `labels` was drawn from --
    the caller is responsible for mapping them back to global indices if
    `labels` is itself a subset). Returns (train_idx, val_idx), each a sorted
    list of indices into `labels`."""
    rng = seeded_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[int(lab)].append(idx)

    train_idx, val_idx = [], []
    for lab, idxs in sorted(by_class.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_fraction))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)
