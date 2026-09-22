"""Task 3, Step 3: SAM -- parameter-space stability.

min_theta max_{||eps||_2 <= rho} L_ERM(theta + eps)

Each source batch: (1) forward/backward at theta -> first_step() ascends to
the nearby worst-case point; (2) a SECOND forward/backward at that perturbed
point -> second_step() restores theta and applies the real AdamW update
using the gradient computed at the perturbed point. Two forward/backward
passes per step, same frozen-BatchNorm policy as every other method.
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
from domain_metrics import evaluate_source_domains  # noqa: E402
from sam_optimizer import SAM  # noqa: E402


def train_sam(
    model, source_iterators: dict, val_datasets: dict, device: str,
    transform_eval, rho: float = 0.05, max_epochs: int = 30, steps_per_epoch: int = 50,
    lr: float = 1e-4, weight_decay: float = 1e-4, patience: int = 5,
):
    """Returns (best_model, history)."""
    model = model.to(device)
    optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=rho, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"cls_loss": [], "val_mean_macro_f1": [], "val_per_domain": []}

    for epoch in range(max_epochs):
        model.train()
        freeze_batchnorm_running_stats(model)
        epoch_cls_loss = 0.0

        for step in range(steps_per_epoch):
            imgs_list, labels_list = [], []
            for domain, loader in source_iterators.items():
                imgs, labels = loader.next_batch()
                imgs_list.append(imgs)
                labels_list.append(labels)
            imgs = torch.cat(imgs_list, dim=0).to(device)
            labels = torch.cat(labels_list, dim=0).to(device)

            # First pass: gradient at theta, then ascend to the worst-case
            # point within radius rho.
            logits1 = model(imgs)
            loss1 = criterion(logits1, labels)
            loss1.backward()
            optimizer.first_step(zero_grad=True)

            # Second pass: gradient AT the perturbed point (same batch,
            # frozen BN policy applies here too -- model is still in
            # train() mode with BN running stats frozen).
            logits2 = model(imgs)
            loss2 = criterion(logits2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            epoch_cls_loss += loss1.item()

        epoch_cls_loss /= steps_per_epoch
        eval_result = evaluate_source_domains(model, val_datasets, device, transform_eval)
        val_f1 = eval_result["mean_macro_f1"]

        history["cls_loss"].append(epoch_cls_loss)
        history["val_mean_macro_f1"].append(val_f1)
        history["val_per_domain"].append(eval_result["per_domain"])

        print(f"  epoch {epoch + 1}/{max_epochs}  cls_loss={epoch_cls_loss:.4f}  "
              f"val_mean_macro_f1={val_f1:.4f}")

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
