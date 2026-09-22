"""Task 3, Step 2: DAN-DG -- pairwise source-domain alignment.

Unlike Task 2's DAN (which aligns source features to UNLABELED TARGET
features), DAN-DG never touches Sketch at all: it applies the same MMD
mechanism to every unordered pair of the three OBSERVED source domains
(Photo-Art, Photo-Cartoon, Art-Cartoon), averaged. Classification loss still
uses all source labels; MMD only pushes the source domains' own marginal
feature distributions to look similar to each other.

L_DAN-DG = L_ERM + (lambda_DG / 3) * sum over unordered source pairs of MMD^2
"""
from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "evaluation"))
from backbone import freeze_batchnorm_running_stats  # noqa: E402
from domain_metrics import evaluate_source_domains  # noqa: E402
from mmd import mmd2  # noqa: E402


def train_dan_dg(
    model, source_iterators: dict, val_datasets: dict, device: str,
    transform_eval, lambda_dg: float = 1.0, max_epochs: int = 30, steps_per_epoch: int = 50,
    lr: float = 1e-4, weight_decay: float = 1e-4, patience: int = 5,
):
    """source_iterators: {domain: InfiniteDomainLoader}, 8/domain/step (24
    total, matching ERM's batch composition -- no target involved at all).
    Returns (best_model, history) with separate classification and MMD loss
    curves."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    domain_names = list(source_iterators.keys())
    pairs = list(itertools.combinations(domain_names, 2))  # 3 unordered pairs

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"cls_loss": [], "mmd_loss": [], "val_mean_macro_f1": [], "val_per_domain": []}

    for epoch in range(max_epochs):
        model.train()
        freeze_batchnorm_running_stats(model)
        epoch_cls_loss, epoch_mmd_loss = 0.0, 0.0

        for step in range(steps_per_epoch):
            batch_imgs, batch_labels = {}, {}
            for domain, loader in source_iterators.items():
                imgs, labels = loader.next_batch()
                batch_imgs[domain] = imgs.to(device)
                batch_labels[domain] = labels.to(device)

            all_imgs = torch.cat(list(batch_imgs.values()), dim=0)
            all_labels = torch.cat(list(batch_labels.values()), dim=0)

            optimizer.zero_grad()
            all_feats = model.forward_features(all_imgs)
            all_logits = model.classify_features(all_feats)
            cls_loss = criterion(all_logits, all_labels)

            # split features back out per domain for pairwise MMD
            sizes = [batch_imgs[d].size(0) for d in domain_names]
            feats_by_domain = dict(zip(domain_names, torch.split(all_feats, sizes)))

            mmd_loss = 0.0
            for a, b in pairs:
                mmd_loss = mmd_loss + mmd2(feats_by_domain[a], feats_by_domain[b])
            mmd_loss = mmd_loss / len(pairs)

            loss = cls_loss + lambda_dg * mmd_loss
            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_mmd_loss += float(mmd_loss.item() if torch.is_tensor(mmd_loss) else mmd_loss)

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
