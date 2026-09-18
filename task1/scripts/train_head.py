"""Train a LinearHead on frozen features, with early stopping on validation
accuracy, per the assignment's specified recipe: AdamW, lr=1e-3, wd=1e-4,
<=50 epochs, early stop after 5 epochs without val-accuracy improvement.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.metrics import top1_accuracy  # noqa: E402


def train_linear_head(
    head: nn.Module,
    train_feats: torch.Tensor, train_labels: torch.Tensor,
    val_feats: torch.Tensor, val_labels: torch.Tensor,
    max_epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-4,
    patience: int = 5, batch_size: int = 128, device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """Returns (best_head, history). `best_head` has the weights from the
    epoch with the highest validation accuracy (early-stopping checkpoint)."""
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True
    )

    best_val_acc = -1.0
    best_state = copy.deepcopy(head.state_dict())
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(max_epochs):
        head.train()
        epoch_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * feats.size(0)
        epoch_loss /= len(train_feats)

        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats.to(device))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = top1_accuracy(val_preds, val_labels.numpy())

        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  early stop at epoch {epoch + 1} "
                      f"(best val acc {best_val_acc:.4f})")
                break

    head.load_state_dict(best_state)
    history["best_val_acc"] = best_val_acc
    history["epochs_trained"] = epoch + 1
    return head, history