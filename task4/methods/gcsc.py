"""Task 4, Step 3: GCSC -- strong closed-set classifier.

Identical to Vanilla in every respect (initialization, optimizer, schedule,
batch size, epochs, seed, checkpoint-selection rule) EXCEPT that
RandAugment(num_ops=2, magnitude=9) is inserted into the training transform
after the crop+flip and before ToTensor/Normalize. Reuses vanilla.py's
train_vanilla() training loop unchanged -- the only difference is which
transform the training dataset uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
from cifar10 import CIFAR_MEAN, CIFAR_STD  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from vanilla import train_vanilla  # noqa: E402

GCSC_TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def train_gcsc(model, train_ds, val_ds, device: str, **kwargs):
    """train_ds must already be constructed with GCSC_TRAIN_TRANSFORM
    (swap it in via CIFARSubset's `.transform` attribute or by rebuilding
    the split with this transform) -- see task4/train.py."""
    return train_vanilla(model, train_ds, val_ds, device, **kwargs)
