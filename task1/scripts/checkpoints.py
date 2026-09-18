"""Save/load helpers so later Task 1 steps (color bias, translation, patch
shuffle) can reuse the exact trained heads and clean predictions from the
clean-baseline step instead of retraining or recomputing them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_head(head: torch.nn.Module, results_dir: str, backbone_name: str) -> Path:
    ckpt_dir = Path(results_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{backbone_name}_head.pt"
    torch.save(head.state_dict(), path)
    return path


def load_head_state(results_dir: str, backbone_name: str) -> dict:
    path = Path(results_dir) / "checkpoints" / f"{backbone_name}_head.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved head checkpoint at {path}. Run the clean baseline step first."
        )
    return torch.load(path, map_location="cpu")


def save_predictions(results_dir: str, tag: str, preds: np.ndarray,
                      probs: np.ndarray, labels: np.ndarray) -> Path:
    """tag examples: 'clean_resnet50', 'clean_clip_zero_shot'."""
    pred_dir = Path(results_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{tag}.npz"
    np.savez(path, preds=preds, probs=probs, labels=labels)
    return path


def load_predictions(results_dir: str, tag: str) -> dict:
    path = Path(results_dir) / "predictions" / f"{tag}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved predictions at {path}. Run the clean baseline step first."
        )
    data = np.load(path)
    return {"preds": data["preds"], "probs": data["probs"], "labels": data["labels"]}