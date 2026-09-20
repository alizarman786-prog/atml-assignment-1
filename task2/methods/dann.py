"""Task 2, Step 3: DANN -- adversarial alignment via gradient reversal.

Only source examples contribute to the classification loss; both source and
target examples contribute to the domain-classification loss (source=0,
target=1). Loss = cls_loss + domain_loss (unit weight). The gradient
reversal layer sends the domain discriminator's gradient back through the
shared feature extractor with its sign flipped and scaled by alpha(p),
which ramps from 0 to max_alpha over training progress p -- early in
training the representation can still learn the class task cleanly; later,
the growing reversed gradient increasingly pushes the features toward
domain confusion.
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
from domain_discriminator import DomainDiscriminator  # noqa: E402
from metrics import evaluate_source_domains  # noqa: E402
from grl import gradient_reversal, dann_alpha  # noqa: E402


def train_dann(
    model, source_iterators: dict, target_iterator, val_datasets: dict, device: str,
    transform_eval, max_epochs: int = 30, steps_per_epoch: int = 50,
    lr: float = 1e-4, weight_decay: float = 1e-4, patience: int = 5,
    max_alpha: float = 1.0,
):
    """Returns (best_model, discriminator, history). The discriminator is
    also returned since some analyses (e.g. inspecting domain-confusion
    behavior) may want it, though only the backbone+head checkpoint is
    saved for the main comparison."""
    model = model.to(device)
    discriminator = DomainDiscriminator(input_dim=model.feature_dim).to(device)

    # Separate optimizers for the classifier network and the discriminator,
    # each with the SAME AdamW hyperparameters (lr, weight_decay) -- this
    # does not deviate from the assignment's fixed-optimizer requirement
    # (the classifier's own training settings are unchanged from
    # Source-only/DAN); it only decouples Adam's per-network moment
    # estimates, which is a standard adversarial-training practice. A single
    # shared optimizer was found empirically to let the two networks'
    # gradient statistics contaminate each other, producing persistent
    # instability once adversarial pressure ramped up.
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=weight_decay)
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    total_steps = max_epochs * steps_per_epoch
    global_step = 0

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = {"cls_loss": [], "domain_loss": [], "alpha": [], "val_mean_macro_f1": [], "val_per_domain": []}

    for epoch in range(max_epochs):
        model.train()
        discriminator.train()
        freeze_batchnorm_running_stats(model)
        epoch_cls_loss, epoch_domain_loss, epoch_alpha = 0.0, 0.0, 0.0

        for step in range(steps_per_epoch):
            p = global_step / total_steps
            alpha = dann_alpha(p, max_alpha=max_alpha)

            src_imgs_list, src_labels_list = [], []
            for domain, loader in source_iterators.items():
                imgs, labels = loader.next_batch()
                src_imgs_list.append(imgs)
                src_labels_list.append(labels)
            src_imgs = torch.cat(src_imgs_list, dim=0).to(device)
            src_labels = torch.cat(src_labels_list, dim=0).to(device)
            n_src = src_imgs.size(0)

            tgt_imgs, _tgt_labels_unused = target_iterator.next_batch()
            tgt_imgs = tgt_imgs.to(device)
            n_tgt = tgt_imgs.size(0)

            model_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            src_feats = model.forward_features(src_imgs)
            src_logits = model.classify_features(src_feats)
            cls_loss = cls_criterion(src_logits, src_labels)

            tgt_feats = model.forward_features(tgt_imgs)

            combined_feats = torch.cat([src_feats, tgt_feats], dim=0)
            reversed_feats = gradient_reversal(combined_feats, alpha)
            domain_logits = discriminator(reversed_feats)
            domain_labels = torch.cat([
                torch.zeros(n_src, dtype=torch.long, device=device),
                torch.ones(n_tgt, dtype=torch.long, device=device),
            ])
            domain_loss = domain_criterion(domain_logits, domain_labels)

            loss = cls_loss + domain_loss

            # Skip the update entirely for anomalous batches -- not just
            # literal NaN/Inf, but any loss far outside the normal ~0.5-2
            # range this training run otherwise operates in. Once Adam's
            # internal moment estimates get poisoned by even one such batch,
            # every future step is corrupted too (the moving average never
            # self-corrects), producing exactly the erratic, non-monotonic
            # blowups observed empirically -- gradient clipping alone cannot
            # fix this because it acts after the fact, on an
            # already-unstable optimizer state. Catching the bad batch
            # BEFORE backward() (and logging it, not silently ignoring it)
            # is the standard remedy.
            if not torch.isfinite(loss) or loss.item() > 20.0:
                print(f"  [warning] anomalous loss={loss.item():.2f} at epoch {epoch + 1} "
                      f"step {step}, skipping update")
                model_optimizer.zero_grad()
                disc_optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                model_optimizer.step()
                disc_optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_alpha += alpha
            global_step += 1

        epoch_cls_loss /= steps_per_epoch
        epoch_domain_loss /= steps_per_epoch
        epoch_alpha /= steps_per_epoch
        eval_result = evaluate_source_domains(model, val_datasets, device, transform_eval)
        val_f1 = eval_result["mean_macro_f1"]

        history["cls_loss"].append(epoch_cls_loss)
        history["domain_loss"].append(epoch_domain_loss)
        history["alpha"].append(epoch_alpha)
        history["val_mean_macro_f1"].append(val_f1)
        history["val_per_domain"].append(eval_result["per_domain"])

        print(f"  epoch {epoch + 1}/{max_epochs}  cls_loss={epoch_cls_loss:.4f}  "
              f"domain_loss={epoch_domain_loss:.4f}  alpha={epoch_alpha:.3f}  "
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
    return model, discriminator, history
