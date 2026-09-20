"""Maximum Mean Discrepancy (MMD) with a multi-kernel RBF, used by DAN
(Task 2, Step 2) and DAN-DG (Task 3). Bandwidths are set per-batch as a
fixed multiple of the median pairwise squared distance, per the assignment.
"""
from __future__ import annotations

import torch


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between every row of x and every row of y.
    x: (n, d), y: (m, d) -> (n, m)."""
    x_sq = (x ** 2).sum(dim=1, keepdim=True)          # (n, 1)
    y_sq = (y ** 2).sum(dim=1, keepdim=True).T          # (1, m)
    return x_sq + y_sq - 2.0 * x @ y.T


def multi_kernel_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth_multipliers=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """Sum of RBF kernels evaluated on the combined [x; y] batch, with
    bandwidths set as `bandwidth_multipliers` times the median pairwise
    squared distance of the CURRENT combined batch (per the assignment).
    Returns the full (n+m, n+m) kernel matrix (used internally by mmd2)."""
    combined = torch.cat([x, y], dim=0)
    sq_dists = _pairwise_sq_dists(combined, combined)

    # Median of the off-diagonal pairwise squared distances (exclude the
    # zero diagonal, which would bias the median toward zero).
    n = sq_dists.size(0)
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=sq_dists.device)
    median_sq_dist = sq_dists[off_diag_mask].median().clamp(min=1e-8)

    kernel_sum = torch.zeros_like(sq_dists)
    for mult in bandwidth_multipliers:
        bandwidth = mult * median_sq_dist
        kernel_sum = kernel_sum + torch.exp(-sq_dists / (2.0 * bandwidth))
    return kernel_sum


def mmd2(x: torch.Tensor, y: torch.Tensor, bandwidth_multipliers=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """Squared MMD between samples x (n, d) and y (m, d), using a sum of
    RBF kernels with per-batch median-heuristic bandwidths. This is the
    "kernel trick" the assignment refers to: it never constructs the
    feature map phi explicitly, only the kernel matrix."""
    n, m = x.size(0), y.size(0)
    K = multi_kernel_rbf(x, y, bandwidth_multipliers)
    K_xx = K[:n, :n]
    K_yy = K[n:, n:]
    K_xy = K[:n, n:]
    return K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
