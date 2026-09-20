"""Task 2, Step 1: Source-only ERM.

Trains ResNet-18 with cross-entropy over the three labeled source domains
using domain-balanced batches (8 examples per source domain per step, per
the assignment). Checkpoints are selected by mean source-validation
macro-F1, with early stopping. This checkpoint is reused UNCHANGED as the
Task 3 ERM baseline -- do not retrain it there.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "evaluation"))
from backbone import freeze_batchnorm_running_stats  # noqa: E402
from metrics import evaluate_source_domains  # noqa: E402


def train_source_only(
    model, train_iterators: dict, val_datasets: dict, device: str,
    transform_eval, max_epochs: int = 30, steps_per_epoch: int = 50,
    lr: float = 1e-4, weight_decay: float = 1e-4, patience: int = 5,
):
    """train_iterators: {domain: InfiniteDomainLoader} for the 3 source
    domains (already using the training transform internally).
    val_datasets: {domain: PACSSubset} (source validation splits).
    Returns (best_model_state_dict, history)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_mean_macro_f1": [], "val_per_domain": []}

    for epoch in range(max_epochs):
        model.train()
        freeze_batchnorm_running_stats(model)
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            imgs_list, labels_list = [], []
            for domain, loader in train_iterators.items():
                imgs, labels = loader.next_batch()
                imgs_list.append(imgs)
                labels_list.append(labels)
            imgs = torch.cat(imgs_list, dim=0).to(device)
            labels = torch.cat(labels_list, dim=0).to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= steps_per_epoch
        eval_result = evaluate_source_domains(model, val_datasets, device, transform_eval)
        val_f1 = eval_result["mean_macro_f1"]

        history["train_loss"].append(epoch_loss)
        history["val_mean_macro_f1"].append(val_f1)
        history["val_per_domain"].append(eval_result["per_domain"])

        print(f"  epoch {epoch + 1}/{max_epochs}  loss={epoch_loss:.4f}  "
              f"val_mean_macro_f1={val_f1:.4f}  "
              f"per_domain={ {d: round(m['macro_f1'], 3) for d, m in eval_result['per_domain'].items()} }")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stop at epoch {epoch + 1} (best val mean macro-F1 {best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    history["best_val_mean_macro_f1"] = best_val_f1
    history["epochs_trained"] = epoch + 1
    return model, history
