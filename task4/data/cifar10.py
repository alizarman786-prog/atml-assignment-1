"""Task 4: CIFAR-10 (known classes) loading and train/val splitting.

A stratified 90/10 split of the official training partition (seed 6304) is
used for training/checkpoint-selection; the complete official test set is
reserved for final known-class evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.splits import stratified_split  # noqa: E402

SEED = 6304
VAL_FRACTION = 0.1

# Per the assignment: random crop 32x32 with 4px padding + random horizontal
# flip for training; no crop/flip for eval. GCSC additionally inserts
# RandAugment between these two steps (see gcsc.py) -- kept as a separate
# transform there so vanilla/GCSC share this exact base recipe otherwise.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

EVAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])


class CIFARSubset(Dataset):
    """Wraps a torchvision CIFAR dataset restricted to a fixed list of
    indices, with a swappable transform (so the SAME underlying images can
    be used with the train transform during training and the eval
    transform during evaluation, without duplicating data)."""

    def __init__(self, base_dataset, indices: list[int], transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        return self.transform(img), label


def load_cifar10_splits(data_root: str, train_transform=TRAIN_TRANSFORM, eval_transform=EVAL_TRANSFORM):
    """Returns (train_ds, val_ds, test_ds). train/val come from a stratified
    90/10 split (seed 6304) of the official training partition; test is the
    complete official CIFAR-10 test set."""
    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
    test_full = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)

    labels = train_full.targets
    train_idx, val_idx = stratified_split(labels, VAL_FRACTION, SEED)

    train_ds = CIFARSubset(train_full, train_idx, train_transform)
    val_ds = CIFARSubset(train_full, val_idx, eval_transform)
    test_ds = CIFARSubset(test_full, list(range(len(test_full))), eval_transform)
    return train_ds, val_ds, test_ds


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
