"""Dataset wrapper for Task 1.

Wraps torchvision STL-10 / Oxford-IIIT Pets, restricted to a fixed list of
indices (train_idx / val_idx / eval_subset_idx from make_subset.py's output),
and returns images resized to the shared 224x224 canvas (via
transforms.to_base_224) BEFORE any backbone-specific preprocessing --
consistent with the assignment's requirement that every model receive the
same clean/transformed images built on a common 224x224 RGB image.

Backbone-specific normalization is applied separately, by the caller, using
each backbone's own `.preprocess` on top of the PIL image this dataset
returns.
"""
from __future__ import annotations

import json
from pathlib import Path

import torchvision
from torch.utils.data import Dataset

from transforms import to_base_224  # sibling module in task1/data/


class Task1Dataset(Dataset):
    """Returns (PIL.Image at 224x224 RGB, int label) for the given indices
    into the official torchvision train or test partition."""

    def __init__(self, dataset_name: str, data_root: str, official_split: str,
                 indices: list[int]):
        assert official_split in ("train", "test")
        self.dataset_name = dataset_name
        self.indices = indices

        if dataset_name == "stl10":
            self._ds = torchvision.datasets.STL10(
                root=data_root, split=official_split, download=True
            )
        elif dataset_name == "pets":
            tv_split = "trainval" if official_split == "train" else "test"
            self._ds = torchvision.datasets.OxfordIIITPet(
                root=data_root, split=tv_split, target_types="category", download=True
            )
        else:
            raise ValueError(f"Unknown dataset {dataset_name!r}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        real_idx = self.indices[i]
        img, label = self._ds[real_idx]
        img = img.convert("RGB")
        img = to_base_224(img)
        return img, int(label)

    @property
    def class_names(self) -> list[str]:
        if self.dataset_name == "stl10":
            return list(self._ds.classes)
        return list(self._ds.classes)


def load_splits(splits_path: str) -> dict:
    return json.loads(Path(splits_path).read_text())


def build_split_datasets(dataset_name: str, data_root: str, splits_path: str):
    """Returns (train_ds, val_ds, eval_ds) built from a make_subset.py split file.
    train/val come from the official train partition; eval comes from the
    official test partition."""
    splits = load_splits(splits_path)
    assert splits["dataset"] == dataset_name, (
        f"splits file is for {splits['dataset']!r}, not {dataset_name!r}"
    )
    train_ds = Task1Dataset(dataset_name, data_root, "train", splits["train_idx"])
    val_ds = Task1Dataset(dataset_name, data_root, "train", splits["val_idx"])
    eval_ds = Task1Dataset(dataset_name, data_root, "test", splits["eval_subset_idx"])
    return train_ds, val_ds, eval_ds