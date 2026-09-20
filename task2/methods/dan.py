"""Task 2, Step 2: DAN -- MMD alignment.

L_DAN = L_cls + lambda_MMD * MMD^2(source_features, target_features)

Source batches remain domain-balanced (8/domain/step, 24 total); the target
batch is 24 unlabeled Sketch images per step, giving equal source/target
batch sizes per the assignment. MMD is applied to the 512-d feature
immediately before the classifier head.
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
from mmd import mmd2  # noqa: E402


def train_dan(
    model, source_iterators: dict, target_iterator, val_datasets: dict, device: str,
    transform_eval, lambda_mmd: float = 1.0, max_epochs: int = 30, steps_per_epoch: int = 50,
    lr: float = 1e-4, weight_decay: float = 1e-4, patience: int = 5,
):
    """source_iterators: {domain: InfiniteDomainLoader}, 8/domain/step.
    target_iterator: InfiniteDomainLoader over the full (unlabeled-in-spirit
    -- labels are present in the batch but simply unused) target domain,
    24/step. Returns (best_model, history) with separate classification and
    MMD loss curves for the required alignment-diagnostic evidence."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"cls_loss": [], "mmd_loss": [], "val_mean_macro_f1": [], "val_per_domain": []}

    for epoch in range(max_epochs):
        model.train()
        freeze_batchnorm_running_stats(model)
        epoch_cls_loss, epoch_mmd_loss = 0.0, 0.0

        for step in range(steps_per_epoch):
            src_imgs_list, src_labels_list = [], []
            for domain, loader in source_iterators.items():
                imgs, labels = loader.next_batch()
                src_imgs_list.append(imgs)
                src_labels_list.append(labels)
            src_imgs = torch.cat(src_imgs_list, dim=0).to(device)
            src_labels = torch.cat(src_labels_list, dim=0).to(device)

            tgt_imgs, _tgt_labels_unused = target_iterator.next_batch()
            tgt_imgs = tgt_imgs.to(device)

            optimizer.zero_grad()
            src_feats = model.forward_features(src_imgs)
            src_logits = model.classify_features(src_feats)
            cls_loss = criterion(src_logits, src_labels)

            tgt_feats = model.forward_features(tgt_imgs)
            mmd_loss = mmd2(src_feats, tgt_feats)

            loss = cls_loss + lambda_mmd * mmd_loss
            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_mmd_loss += mmd_loss.item()

        epoch_cls_loss /= steps_per_epoch
        epoch_mmd_loss /= steps_per_epoch
        eval_result = evaluate_source_domains(model, val_datasets, device, transform_eval)
        val_f1 = eval_result["mean_macro_f1"]

        history["cls_loss"].append(epoch_cls_loss)
        history["mmd_loss"].append(epoch_mmd_loss)
        history["val_mean_macro_f1"].append(val_f1)
        history["val_per_domain"].append(eval_result["per_domain"])

        print(f"  epoch {epoch + 1}/{max_epochs}  cls_loss={epoch_cls_loss:.4f}  "
              f"mmd_loss={epoch_mmd_loss:.4f}  val_mean_macro_f1={val_f1:.4f}")

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
