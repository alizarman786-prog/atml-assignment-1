"""Task 3 Step 4: local sharpness proxy, used identically for ERM, DAN-DG,
and SAM so their local loss-landscape stability is directly comparable.

Delta_sharp = L(theta + epsilon) - L(theta),
    epsilon = rho * grad_L(theta) / ||grad_L(theta)||_2

Measured on ONE fixed validation batch (32 examples per source domain, seed
6304), with the model in eval mode. A single normalized gradient-ascent step
of radius rho=0.05 (matching SAM's rho for the main comparison) -- this is a
standardized LOCAL diagnostic, not a claim about the entire loss landscape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from backbone import freeze_batchnorm_running_stats  # noqa: E402


def _sample_fixed_batch(source_datasets: dict, transform, n_per_domain: int, seed: int, device: str):
    rng = np.random.default_rng(seed)
    imgs_list, labels_list = [], []
    for domain, ds in source_datasets.items():
        idx = rng.choice(len(ds), size=min(n_per_domain, len(ds)), replace=False)
        for i in idx:
            img, label = ds[int(i)]
            imgs_list.append(transform(img))
            labels_list.append(label)
    imgs = torch.stack(imgs_list).to(device)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)
    return imgs, labels


def sharpness_proxy(
    model, source_train_datasets: dict, transform, device: str,
    rho: float = 0.05, n_per_domain: int = 32, seed: int = 6304,
) -> dict:
    """source_train_datasets: {domain: Dataset} -- the assignment specifies
    sampling this fixed batch from source data (validation-batch-sized, 32
    per domain); uses the TRAIN split pool since that's what's available at
    this stage of the pipeline without touching Sketch."""
    imgs, labels = _sample_fixed_batch(source_train_datasets, transform, n_per_domain, seed, device)

    model.eval()
    freeze_batchnorm_running_stats(model)  # no-op in eval mode, kept for clarity
    criterion = nn.CrossEntropyLoss()

    # L(theta)
    model.zero_grad()
    logits = model(imgs)
    loss_theta = criterion(logits, labels)
    loss_theta.backward()

    # epsilon = rho * grad / ||grad||_2 (global norm across all parameters)
    params = [p for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
    scale = rho / (grad_norm + 1e-12)

    with torch.no_grad():
        originals = []
        for p in params:
            originals.append(p.data.clone())
            p.data.add_(p.grad * scale)

    # L(theta + epsilon)
    model.zero_grad()
    with torch.no_grad():
        logits_perturbed = model(imgs)
        loss_perturbed = criterion(logits_perturbed, labels)

    # restore original parameters
    with torch.no_grad():
        for p, orig in zip(params, originals):
            p.data.copy_(orig)
    model.zero_grad()

    delta_sharp = float(loss_perturbed.item() - loss_theta.item())
    return {
        "rho": rho,
        "loss_theta": float(loss_theta.item()),
        "loss_perturbed": float(loss_perturbed.item()),
        "delta_sharp": delta_sharp,
    }
