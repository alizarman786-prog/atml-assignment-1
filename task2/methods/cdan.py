"""Task 2, Step 4: CDAN -- class-conditional adversarial alignment.

Identical to DANN except the domain discriminator sees g(x) = vec(f (x)
p) -- the outer product of the 512-d feature and the classifier's
num_classes-d softmax probability vector, flattened -- instead of the raw
feature alone. No entropy conditioning; f and p are NOT detached (per the
assignment), so the adversarial domain loss's gradient flows back through
both the feature extractor and (via p) the classifier head itself.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "evaluation"))
from backbone import freeze_batchnorm_running_stats  # noqa: E402
from domain_discriminator import DomainDiscriminator  # noqa: E402
from metrics import evaluate_source_domains  # noqa: E402
from grl import gradient_reversal, dann_alpha  # noqa: E402


def multilinear_map(feat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """g(x) = vec(f (x) p): outer product of feature (N, D) and class
    probabilities (N, C), flattened per-example to (N, D*C)."""
    n, d = feat.shape
    c = probs.shape[1]
    outer = feat.unsqueeze(2) * probs.unsqueeze(1)   # (N, D, C)
    return outer.view(n, d * c)


def train_cdan(
    model, source_iterators: dict, target_iterator, val_datasets: dict, device: str,
    transform_eval, num_classes: int, max_epochs: int = 30, steps_per_epoch: int = 50,
    lr: float = 1e-4, weight_decay: float = 1e-4, patience: int = 5,
    max_alpha: float = 1.0,
):
    """Returns (best_model, discriminator, history)."""
    model = model.to(device)
    discriminator = DomainDiscriminator(input_dim=model.feature_dim * num_classes).to(device)

    # Separate optimizers -- see dann.py's identical comment for reasoning.
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
            p_progress = global_step / total_steps
            alpha = dann_alpha(p_progress, max_alpha=max_alpha)

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
            tgt_logits = model.classify_features(tgt_feats)

            # Softmax class-probability vectors for BOTH source and target
            # (target has no labels, but CDAN only needs the classifier's
            # own predicted distribution, not ground truth).
            src_probs = F.softmax(src_logits, dim=1)
            tgt_probs = F.softmax(tgt_logits, dim=1)

            src_g = multilinear_map(src_feats, src_probs)
            tgt_g = multilinear_map(tgt_feats, tgt_probs)
            combined_g = torch.cat([src_g, tgt_g], dim=0)

            reversed_g = gradient_reversal(combined_g, alpha)
            domain_logits = discriminator(reversed_g)
            domain_labels = torch.cat([
                torch.zeros(n_src, dtype=torch.long, device=device),
                torch.ones(n_tgt, dtype=torch.long, device=device),
            ])
            domain_loss = domain_criterion(domain_logits, domain_labels)

            loss = cls_loss + domain_loss

            # Same anomalous-loss skip as DANN -- see that file's comment
            # for the reasoning.
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
