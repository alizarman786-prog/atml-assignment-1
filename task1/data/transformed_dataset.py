"""Wraps any Task1Dataset-like dataset (returning PIL images at 224x224) and
applies a given transform function to each image before returning it. Used
for color bias, and reusable for any other pixel-space intervention that
takes a single PIL image and returns a PIL image.
"""
from __future__ import annotations

from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform_fn):
        """transform_fn: Callable[[PIL.Image], PIL.Image]. Applied on top of
        the base dataset's already-224x224 image, before any backbone
        preprocessing."""
        self.base = base_dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return self.transform_fn(img), label

    @property
    def class_names(self):
        return self.base.class_names