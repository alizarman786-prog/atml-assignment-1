"""PACS experimental protocol shared by Task 2 and Task 3: stratified
per-domain train/val splits, and domain-balanced batch iteration (every
training step pulls a fixed number of examples from each domain, so no
domain dominates a batch by virtue of having more images).

Both tasks use Photo, Art Painting, Cartoon as labeled sources and Sketch as
the (adapted-to, in Task 2; withheld, in Task 3) target -- see each task's
own README for what may/may not touch Sketch during training.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pacs import PACSSubset, get_domain_indices, get_domain_labels  # noqa: E402
from common.splits import stratified_split  # noqa: E402

SEED = 6304
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
VAL_FRACTION = 0.2


def build_pacs_splits(hf_dataset, out_path: str | None = None) -> dict:
    """Stratified 80/20 split of each source domain (seed 6304), plus the
    full target domain index list (unlabeled for Task 2's adaptation,
    withheld entirely from training in Task 3). Returns a dict; if
    `out_path` is given, also writes it as JSON (global indices only, no
    images duplicated) for reuse across Task 2 and Task 3 without
    re-splitting."""
    splits = {"seed": SEED, "source_domains": {}, "target_domain": TARGET_DOMAIN}

    for domain in SOURCE_DOMAINS:
        global_idx = get_domain_indices(hf_dataset, domain)
        labels = get_domain_labels(hf_dataset, global_idx)
        # stratified_split returns indices INTO `labels` (0..len-1); map back
        # to global hf_dataset indices before saving.
        local_train, local_val = stratified_split(labels, VAL_FRACTION, SEED)
        splits["source_domains"][domain] = {
            "train_idx": [global_idx[i] for i in local_train],
            "val_idx": [global_idx[i] for i in local_val],
        }

    splits["target_idx"] = get_domain_indices(hf_dataset, TARGET_DOMAIN)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(splits, indent=2))
    return splits


def load_pacs_splits(path: str) -> dict:
    return json.loads(Path(path).read_text())


def build_source_datasets(hf_dataset, splits: dict) -> dict:
    """Returns {domain: {"train": PACSSubset, "val": PACSSubset}} for each
    source domain."""
    out = {}
    for domain, idx in splits["source_domains"].items():
        out[domain] = {
            "train": PACSSubset(hf_dataset, idx["train_idx"]),
            "val": PACSSubset(hf_dataset, idx["val_idx"]),
        }
    return out


def _collate_with_transform(batch, transform):
    imgs, labels = zip(*batch)
    tensors = torch.stack([transform(img) for img in imgs])
    labels = torch.tensor(labels, dtype=torch.long)
    return tensors, labels


class InfiniteDomainLoader:
    """Wraps one domain's DataLoader so `next_batch()` always returns a
    batch, cycling back to the start (with a fresh shuffle) when the
    underlying data is exhausted -- standard pattern for domain-balanced
    training where domains have different sizes and/or per-step batch
    counts don't evenly divide the dataset."""

    def __init__(self, dataset, batch_size: int, transform, num_workers: int = 2):
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers,
            collate_fn=lambda b: _collate_with_transform(b, transform),
        )
        self._iter = iter(self.loader)

    def next_batch(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


def build_domain_iterators(domain_datasets: dict, batch_size_per_domain: int,
                            transform, num_workers: int = 2) -> dict:
    """domain_datasets: {domain_name: Dataset}. Returns {domain_name:
    InfiniteDomainLoader} for pulling per-domain balanced batches every
    training step."""
    return {
        name: InfiniteDomainLoader(ds, batch_size_per_domain, transform, num_workers)
        for name, ds in domain_datasets.items()
    }
