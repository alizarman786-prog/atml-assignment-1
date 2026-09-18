"""A dataset backed by a plain in-memory list of (PIL.Image, label) pairs.
Used when a transform must be precomputed once and then reused identically
across multiple backbones (e.g. patch shuffle, where the assignment requires
"exactly the same shuffled images" for every model) rather than recomputed
per-backbone via a transform function (which is fine for translation/color,
where the transform is a pure deterministic function of the image alone).
"""
from __future__ import annotations

from torch.utils.data import Dataset


class InMemoryDataset(Dataset):
    def __init__(self, items: list, class_names: list[str]):
        self.items = items
        self._class_names = class_names

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    @property
    def class_names(self):
        return self._class_names
