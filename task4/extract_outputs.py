"""Task 4: extracts and caches penultimate features + logits for a trained
checkpoint, on CIFAR-10 train/val/test and the fixed CIFAR-100 near/far
unknown sets. All four post-hoc scores are computed from these SAME cached
arrays, per the assignment's requirement that "all four scores must use
exactly the same saved logits and features."

Train features are extracted WITHOUT augmentation (eval transform), per the
assignment's Mahalanobis fitting requirement ("unaugmented CIFAR-10 training
features").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent / "data"))
sys.path.append(str(Path(__file__).resolve().parent / "models"))
from cifar10 import load_cifar10_splits, EVAL_TRANSFORM  # noqa: E402
from cifar100_unknowns import load_near_far_unknowns  # noqa: E402
from resnet_cifar import build_cifar_resnet18  # noqa: E402


@torch.no_grad()
def extract_features_and_logits(model, dataset, device: str, batch_size: int = 256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_feats, all_logits, all_labels = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = model.forward_features(imgs)
        logits = model.classify_features(feats)
        all_feats.append(feats.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_labels.append(np.asarray(labels))
    return (
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def extract_all_outputs(checkpoint_path: str, data_root: str, device: str,
                         num_classes: int = 10, cache_dir: str = "cache", tag: str = "vanilla"):
    """Loads `checkpoint_path` into a fresh CIFARResNet18, extracts
    features+logits for CIFAR-10 train (unaugmented)/val/test and CIFAR-100
    near/far unknowns, and saves everything to `cache_dir/{tag}_*.npz`.

    Handles PROSER's extended head automatically: if the checkpoint's fc
    layer has more outputs than `num_classes` (i.e. it includes dummy
    classes), the model's fc layer is resized to match BEFORE loading, so
    the resulting cached logits correctly include the dummy-class columns
    (needed by evaluate_osr.py's proser_placeholder_score)."""
    model = build_cifar_resnet18(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location=device)

    ckpt_fc_out = state_dict["net.fc.weight"].shape[0]
    if ckpt_fc_out != num_classes:
        print(f"  checkpoint has {ckpt_fc_out} output classes (expected {num_classes}); "
              f"resizing fc layer to match (this is expected for PROSER's dummy classes)")
        model.net.fc = torch.nn.Linear(model.net.fc.in_features, ckpt_fc_out)

    model.load_state_dict(state_dict)
    model = model.to(device)

    train_ds, val_ds, test_ds = load_cifar10_splits(data_root, train_transform=EVAL_TRANSFORM)
    near_ds, far_ds = load_near_far_unknowns(data_root)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds),
                            ("near", near_ds), ("far", far_ds)]:
        feats, logits, labels = extract_features_and_logits(model, ds, device)
        out_path = cache_dir / f"{tag}_{split_name}.npz"
        np.savez(out_path, features=feats, logits=logits, labels=labels)
        print(f"  cached {split_name}: features={feats.shape} logits={logits.shape} -> {out_path}")


def load_cached(cache_dir: str, tag: str, split_name: str) -> dict:
    path = Path(cache_dir) / f"{tag}_{split_name}.npz"
    data = np.load(path)
    return {"features": data["features"], "logits": data["logits"], "labels": data["labels"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="e.g. vanilla, gcsc, proser")
    ap.add_argument("--data-root", default="./data_raw")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[extract_outputs] extracting for tag={args.tag!r} from {args.checkpoint}")
    extract_all_outputs(args.checkpoint, args.data_root, args.device, cache_dir=args.cache_dir, tag=args.tag)
    print("[extract_outputs] done")
