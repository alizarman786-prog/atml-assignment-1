"""Task 1 data preparation.

Produces, with seed 6304:
  1. A stratified 80/20 train/val split of the official training partition
     (used to train each backbone's linear classifier head).
  2. A class-balanced 500-image evaluation subset drawn from the official
     test partition (used for every intervention so all models see the same
     images).

Both splits are saved as JSON files of *indices into the underlying
torchvision dataset* (not copies of the images), so nothing large gets
committed to git and the exact same subset can be reloaded deterministically.

Usage:
    python data/make_subset.py --dataset stl10 --data-root ./data --out-dir results/splits
    python data/make_subset.py --dataset pets  --data-root ./data --out-dir results/splits
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))  # repo root
from common.seed import seeded_rng  # noqa: E402

SEED = 6304
EVAL_SUBSET_SIZE = 500


def _labels_for_dataset(dataset_name: str, data_root: str, split: str) -> np.ndarray:
    """Return the integer label for every example in the given torchvision
    split, without loading/decoding images (labels only)."""
    import torchvision

    if dataset_name == "stl10":
        # STL-10 'train' split is the labeled training partition; 'test' is
        # the labeled test partition (this task doesn't use the unlabeled split).
        ds = torchvision.datasets.STL10(root=data_root, split=split, download=True)
        return np.array(ds.labels)
    elif dataset_name == "pets":
        # Oxford-IIIT Pets: torchvision's "trainval"/"test" splits map to the
        # official training/test partitions.
        tv_split = "trainval" if split == "train" else "test"
        ds = torchvision.datasets.OxfordIIITPet(
            root=data_root, split=tv_split, target_types="category", download=True
        )
        return np.array([label for _, label in ds._samples]) if hasattr(ds, "_samples") \
            else np.array([ds[i][1] for i in range(len(ds))])
    else:
        raise ValueError(f"Unknown dataset {dataset_name!r}; expected 'stl10' or 'pets'.")


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Stratified split preserving class proportions. Returns (train_idx, val_idx)."""
    rng = seeded_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[int(lab)].append(idx)

    train_idx, val_idx = [], []
    for lab, idxs in sorted(by_class.items()):
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_fraction))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def class_balanced_subset(labels: np.ndarray, total: int, seed: int) -> tuple[list[int], dict]:
    """Class-balanced selection from `labels`, sized as close to `total` as
    possible. If a class has fewer than the even per-class quota, use all its
    examples and record the imbalance (per assignment instructions)."""
    rng = seeded_rng(seed)
    classes = sorted(set(int(l) for l in labels))
    per_class_target = total // len(classes)

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_class[int(lab)].append(idx)

    selected = []
    shortfall_log = {}
    for c in classes:
        idxs = np.array(by_class[c])
        rng.shuffle(idxs)
        take = min(per_class_target, len(idxs))
        if take < per_class_target:
            shortfall_log[c] = {"available": int(len(idxs)), "target": per_class_target}
        selected.extend(idxs[:take].tolist())

    # If undershooting `total` due to shortfalls, top up from classes with
    # spare examples (still deterministic given the fixed rng draw order above).
    remaining = total - len(selected)
    if remaining > 0:
        leftover_pool = []
        selected_set = set(selected)
        for c in classes:
            for i in by_class[c]:
                if i not in selected_set:
                    leftover_pool.append(i)
        leftover_pool = np.array(leftover_pool)
        rng.shuffle(leftover_pool)
        selected.extend(leftover_pool[:remaining].tolist())

    return sorted(selected), shortfall_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["stl10", "pets"], required=True)
    ap.add_argument("--data-root", default="./data_raw")
    ap.add_argument("--out-dir", default="results/splits")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[make_subset] loading labels for {args.dataset} (train partition)...")
    train_labels = _labels_for_dataset(args.dataset, args.data_root, "train")
    train_idx, val_idx = stratified_split(train_labels, args.val_fraction, SEED)

    print(f"[make_subset] loading labels for {args.dataset} (test partition)...")
    test_labels = _labels_for_dataset(args.dataset, args.data_root, "test")
    eval_idx, shortfall = class_balanced_subset(test_labels, EVAL_SUBSET_SIZE, SEED)

    split_record = {
        "dataset": args.dataset,
        "seed": SEED,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "eval_subset_idx": eval_idx,
        "eval_subset_size_requested": EVAL_SUBSET_SIZE,
        "eval_subset_size_actual": len(eval_idx),
        "eval_class_shortfalls": shortfall,
    }

    out_path = out_dir / f"{args.dataset}_splits_seed{SEED}.json"
    out_path.write_text(json.dumps(split_record, indent=2))

    print(f"[make_subset] train={len(train_idx)} val={len(val_idx)} "
          f"eval_subset={len(eval_idx)} (requested {EVAL_SUBSET_SIZE})")
    if shortfall:
        print(f"[make_subset] WARNING class imbalance in eval subset: {shortfall}")
    print(f"[make_subset] wrote {out_path}")


if __name__ == "__main__":
    main()
