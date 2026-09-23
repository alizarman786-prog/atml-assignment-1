"""Task 4, Step 4: PROSER -- classifier and data placeholders.

Extends the Vanilla checkpoint's 10-class head with 5 additional randomly
initialized "dummy" output units (15 total). Each mini-batch is split in
half: the first half trains CLASSIFIER placeholders (on ordinary CIFAR-10
images), the second half trains DATA placeholders (on manifold-mixup
features between two different-class images).

Classifier placeholder loss (beta=1): standard closed-set CE (using only
the 10 known-class logits) PLUS a second CE term where the true-class
logit is replaced by the single best dummy logit at that same position --
this directly implements the assignment's description ("once the correct
class is excluded, PROSER encourages one of the dummy classifiers to
become the strongest remaining response"): the substituted dummy score
must remain competitive enough to keep position y the argmax.

Data placeholder loss (gamma=0.1): manifold-mixed features (after layer2,
before layer3) between two different-class images are pushed toward a
POOLED dummy-class score (via logsumexp over the 5 dummy logits, forming
one merged "dummy" competitor alongside the 10 known classes) -- i.e. these
proxy unknown representations should be classified as (some) dummy class
rather than any real class.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parent))
from resnet_cifar import build_cifar_resnet18  # noqa: E402
from manifold_mixup import make_different_class_pairs, manifold_mixup  # noqa: E402


def build_proser_model(vanilla_checkpoint_path: str, device: str,
                        num_known: int = 10, num_dummy: int = 5):
    """Loads the Vanilla checkpoint, then extends its fc layer from
    (512 -> num_known) to (512 -> num_known + num_dummy), copying the
    known-class weights over and leaving the dummy rows at their fresh
    random initialization."""
    model = build_cifar_resnet18(num_classes=num_known)
    model.load_state_dict(torch.load(vanilla_checkpoint_path, map_location=device))

    old_fc = model.net.fc
    feat_dim = old_fc.in_features
    new_fc = nn.Linear(feat_dim, num_known + num_dummy)
    with torch.no_grad():
        new_fc.weight[:num_known].copy_(old_fc.weight)
        new_fc.bias[:num_known].copy_(old_fc.bias)
    model.net.fc = new_fc
    model.feature_dim = feat_dim
    model.num_known = num_known
    model.num_dummy = num_dummy
    return model.to(device)


@torch.no_grad()
def evaluate_known_accuracy(model, dataset, device: str, num_known: int, batch_size: int = 256) -> float:
    """Accuracy using ONLY the known-class logits (first num_known), per
    the assignment's checkpoint-selection rule (CIFAR-10 validation
    accuracy, not conflated with dummy-class outputs)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)[:, :num_known]
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def classifier_placeholder_loss(model, imgs, labels, num_known, beta=1.0):
    logits = model(imgs)                        # (B, K+L)
    known_logits = logits[:, :num_known]
    dummy_logits = logits[:, num_known:]

    loss1 = F.cross_entropy(known_logits, labels)

    max_dummy = dummy_logits.max(dim=1).values   # (B,)
    modified = known_logits.clone()
    modified[torch.arange(labels.size(0), device=labels.device), labels] = max_dummy
    loss2 = F.cross_entropy(modified, labels)

    return loss1 + beta * loss2


def data_placeholder_loss(model, imgs, labels, num_known, num_dummy, gamma=0.1):
    partner_idx = make_different_class_pairs(labels)
    feat_a = model.forward_features_upto_layer2(imgs)
    feat_b = feat_a[partner_idx]
    mixed_feat, _lam = manifold_mixup(feat_a, feat_b, alpha=2.0)

    penultimate = model.forward_from_layer3(mixed_feat)
    logits = model.classify_features(penultimate)   # (B, K+L)
    known_logits = logits[:, :num_known]
    dummy_logits = logits[:, num_known:]

    # Pool the L dummy logits into one merged competitor score via
    # logsumexp, forming a (K+1)-dim vector; target = the merged-dummy slot.
    pooled_dummy = torch.logsumexp(dummy_logits, dim=1, keepdim=True)  # (B, 1)
    collapsed = torch.cat([known_logits, pooled_dummy], dim=1)          # (B, K+1)
    target = torch.full((labels.size(0),), num_known, dtype=torch.long, device=labels.device)

    return gamma * F.cross_entropy(collapsed, target)


def proser_placeholder_score(logits: np.ndarray, num_known: int = 10) -> np.ndarray:
    """PROSER's placeholder-based unknownness score: total softmax
    probability mass assigned to the dummy classes (computed within the
    FULL known+dummy softmax, so it directly reflects how much the dummy
    classifiers' responses compete against the known-class ones). Higher =
    more unknown-looking. This directly implements "combining the strongest
    dummy response with the known-class responses" from the assignment."""
    probs = softmax(logits, axis=1)
    return probs[:, num_known:].sum(axis=1)


def train_proser(
    vanilla_checkpoint_path: str, train_ds, val_ds, device: str,
    num_known: int = 10, num_dummy: int = 5, beta: float = 1.0, gamma: float = 0.1,
    max_epochs: int = 50, batch_size: int = 128,
    lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 5e-4,
    seed: int = 6304,
):
    model = build_proser_model(vanilla_checkpoint_path, device, num_known, num_dummy)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        generator=g, drop_last=True,
    )

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = {"cls_placeholder_loss": [], "data_placeholder_loss": [], "val_accuracy": []}

    for epoch in range(max_epochs):
        model.train()
        epoch_cls_loss, epoch_data_loss = 0.0, 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            half = imgs.size(0) // 2
            imgs_cls, labels_cls = imgs[:half], labels[:half]
            imgs_data, labels_data = imgs[half:], labels[half:]

            optimizer.zero_grad()
            loss_cls = classifier_placeholder_loss(model, imgs_cls, labels_cls, num_known, beta)
            loss_data = data_placeholder_loss(model, imgs_data, labels_data, num_known, num_dummy, gamma)
            loss = loss_cls + loss_data
            loss.backward()
            optimizer.step()

            epoch_cls_loss += loss_cls.item()
            epoch_data_loss += loss_data.item()

        scheduler.step()
        epoch_cls_loss /= len(train_loader)
        epoch_data_loss /= len(train_loader)

        val_acc = evaluate_known_accuracy(model, val_ds, device, num_known)
        history["cls_placeholder_loss"].append(epoch_cls_loss)
        history["data_placeholder_loss"].append(epoch_data_loss)
        history["val_accuracy"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{max_epochs}  cls_placeholder_loss={epoch_cls_loss:.4f}  "
                  f"data_placeholder_loss={epoch_data_loss:.4f}  val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

    model.load_state_dict(best_state)
    history["best_val_accuracy"] = best_val_acc
    return model, history
