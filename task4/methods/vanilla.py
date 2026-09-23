"""Task 4, Step 1: Vanilla closed-set baseline.

SGD (lr=0.1, momentum=0.9, weight_decay=5e-4), cosine decay over the full
100-epoch budget, batch size 128, seed 6304. No early stopping -- trains
the complete 100 epochs, but the CHECKPOINT SAVED is whichever epoch had
the highest CIFAR-10 validation accuracy (tracked throughout, not
necessarily the final epoch).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(model, dataset, device: str, batch_size: int = 256) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_vanilla(
    model, train_ds, val_ds, device: str,
    max_epochs: int = 100, batch_size: int = 128,
    lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4,
    seed: int = 6304,
):
    """Returns (best_model, history). Trains the full max_epochs with no
    early stopping; saves the state_dict with the highest val accuracy seen."""
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        generator=g, drop_last=True,
    )

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_accuracy": [], "lr": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= len(train_loader)

        val_acc = evaluate_accuracy(model, val_ds, device)
        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{max_epochs}  loss={epoch_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  (best={best_val_acc:.4f})  lr={history['lr'][-1]:.5f}")

    model.load_state_dict(best_state)
    history["best_val_accuracy"] = best_val_acc
    return model, history
