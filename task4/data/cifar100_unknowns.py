"""Task 4: fixed CIFAR-100 near/far unknown-class subsets.

The near/far groupings are FIXED by the assignment and must not be revised
after seeing results. CIFAR-100's test partition has exactly 100 images per
fine class, so 8 classes/group gives exactly 800 images/group, matching the
assignment's stated group size with no subsampling needed.
"""
from __future__ import annotations

import torchvision
from torch.utils.data import Dataset

from cifar10 import EVAL_TRANSFORM  # sibling module

NEAR_UNKNOWN_CLASSES = [
    "bus", "pickup_truck", "motorcycle", "tractor",
    "wolf", "fox", "leopard", "camel",
]
FAR_UNKNOWN_CLASSES = [
    "bottle", "bowl", "chair", "clock",
    "keyboard", "mushroom", "sunflower", "wardrobe",
]


class CIFAR100UnknownSubset(Dataset):
    """CIFAR-100 test images restricted to a fixed list of fine class names.
    Returns (transformed_image, fine_label_idx), where fine_label_idx is the
    ORIGINAL CIFAR-100 fine label (0..99) -- NOT a valid CIFAR-10 known-class
    label. This is kept (rather than discarded to -1) so failure-case
    inspection can report which real semantic class a false-accepted
    "unknown" image actually belongs to; use `.classes` (below) to map the
    integer back to its name. Never pass these labels to a CIFAR-10
    accuracy computation."""

    def __init__(self, data_root: str, class_names: list[str], transform=EVAL_TRANSFORM):
        base = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
        name_to_idx = {name: i for i, name in enumerate(base.classes)}
        missing = [n for n in class_names if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"Class name(s) not found in CIFAR-100's fine label list: {missing}. "
                f"Available names: {base.classes}"
            )
        target_indices = {name_to_idx[n] for n in class_names}

        self.base = base
        self.classes = base.classes  # index -> fine class name, for reporting
        self.transform = transform
        # Use base.targets (plain list of ints) rather than decoding every
        # image just to inspect its label.
        self.indices = [i for i, label in enumerate(base.targets) if label in target_indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, fine_label = self.base[self.indices[i]]
        return self.transform(img), fine_label


def load_near_far_unknowns(data_root: str, transform=EVAL_TRANSFORM):
    near_ds = CIFAR100UnknownSubset(data_root, NEAR_UNKNOWN_CLASSES, transform)
    far_ds = CIFAR100UnknownSubset(data_root, FAR_UNKNOWN_CLASSES, transform)
    return near_ds, far_ds
