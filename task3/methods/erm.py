"""Task 3, Step 1: ERM baseline.

Per the assignment, this is NOT retrained -- it is exactly the Source-only
ERM checkpoint from Task 2 (same three source domains, same recipe), loaded
unchanged. This module just wraps "load Task 2's checkpoint" so Task 3's
train.py entry point has a uniform interface across all three methods.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch


def load_erm_from_task2(model, task2_checkpoint_path: str, device: str):
    """Loads Task 2's Source-only checkpoint into `model` in place. Returns
    the model (for interface symmetry with the other methods' train_*
    functions, which also return a model)."""
    ckpt_path = Path(task2_checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Task 2's Source-only checkpoint not found at {ckpt_path}. "
            "Run Task 2's train.py with configs/source_only.yaml first."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)


def copy_checkpoint(task2_checkpoint_path: str, task3_checkpoint_path: str):
    """Copies the Task 2 checkpoint file into Task 3's own results dir, so
    Task 3's results are self-contained and don't depend on Task 2's output
    directory still existing at evaluation time."""
    src = Path(task2_checkpoint_path)
    dst = Path(task3_checkpoint_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
