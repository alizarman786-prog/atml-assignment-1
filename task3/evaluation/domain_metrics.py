"""Evaluation utilities shared by every Task 2 (and Task 3) method: run a
model on a PACS domain split and compute accuracy/macro-F1, then summarize
across the three source domains.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.metrics import top1_accuracy, macro_f1  # noqa: E402


@torch.no_grad()
def evaluate_domain(model, dataset, device: str, transform, batch_size: int = 64):
    """Returns (preds, labels) as numpy arrays for one domain split."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: (
            torch.stack([transform(img) for img, _ in b]),
            torch.tensor([lab for _, lab in b]),
        ),
    )
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        preds = F.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate_source_domains(model, val_datasets: dict, device: str, transform) -> dict:
    """val_datasets: {domain_name: Dataset}. Returns per-domain
    {accuracy, macro_f1} plus 'mean' and 'worst' summaries across domains."""
    per_domain = {}
    for domain, ds in val_datasets.items():
        preds, labels = evaluate_domain(model, ds, device, transform)
        per_domain[domain] = {
            "accuracy": top1_accuracy(preds, labels),
            "macro_f1": macro_f1(preds, labels),
        }
    accs = [m["accuracy"] for m in per_domain.values()]
    f1s = [m["macro_f1"] for m in per_domain.values()]
    return {
        "per_domain": per_domain,
        "mean_accuracy": sum(accs) / len(accs),
        "mean_macro_f1": sum(f1s) / len(f1s),
        "worst_accuracy": min(accs),
        "worst_macro_f1": min(f1s),
    }
