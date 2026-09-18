"""Task 1, Step 5: Patch Structure (4x4 patch shuffle).

Generates ONE non-identity 4x4 patch permutation per eval image (seed 6304),
precomputed once so the assignment's requirement to "reuse exactly the same
shuffled images across models" is guaranteed by construction (every backbone
below runs on the exact same in-memory list of shuffled images, not a
re-randomized transform per backbone). Reports accuracy drop and prediction
consistency relative to clean.

Requires run_task1_clean_baseline.py to have been run first (reuses the
saved heads and clean predictions).

Usage (from inside task1/):
    python scripts/run_task1_patch_shuffle.py --config configs/clean_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset import build_split_datasets  # noqa: E402
from in_memory_dataset import InMemoryDataset  # noqa: E402
from transforms import patch_shuffle  # noqa: E402
from backbones import build_backbone, LinearHead  # noqa: E402
from checkpoints import load_head_state, load_predictions, save_predictions  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.metrics import top1_accuracy, macro_f1, mean_max_confidence, prediction_consistency  # noqa: E402

SEED = 6304
GRID = 4


def build_shuffled_dataset(eval_ds) -> InMemoryDataset:
    """Precompute the one shared set of shuffled images, seeded per-image as
    SEED + index so it is deterministic and identical every time this
    function runs (and therefore identical across every backbone's call)."""
    items = []
    for i in range(len(eval_ds)):
        img, label = eval_ds[i]
        shuffled_img, _perm_id = patch_shuffle(img, seed=SEED + i, grid=GRID)
        items.append((shuffled_img, label))
    return InMemoryDataset(items, class_names=eval_ds.class_names)


def evaluate_and_compare(backbone, head_or_none, ds, device, clean_preds, is_zero_shot=False):
    loader = DataLoader(
        ds, batch_size=64, shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([backbone.preprocess(img) for img, _ in batch]),
            torch.tensor([lab for _, lab in batch]),
        ),
    )
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if is_zero_shot:
                probs = backbone.zero_shot_predict(imgs).cpu()
            else:
                feats = backbone.extract_features(imgs)
                probs = F.softmax(head_or_none(feats), dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = probs.argmax(axis=1)
    metrics = {
        "top1_accuracy": top1_accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels),
        "mean_max_confidence": mean_max_confidence(probs),
        "accuracy_delta_vs_clean": top1_accuracy(preds, labels) - top1_accuracy(clean_preds, labels),
        "prediction_consistency_vs_clean": prediction_consistency(preds, clean_preds),
    }
    return metrics, preds, probs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    results_dir = str(Path(cfg["output"]["results_json"]).parent)

    _, _, eval_ds = build_split_datasets(cfg["dataset"], cfg["data_root"], cfg["splits_path"])
    class_names = eval_ds.class_names

    print("[patch_shuffle] precomputing shared shuffled image set...")
    shuffled_ds = build_shuffled_dataset(eval_ds)

    results = {"dataset": cfg["dataset"], "seed": cfg["seed"], "grid": GRID, "models": {}}

    for name in cfg["backbones"]:
        print(f"\n[patch_shuffle] === {name} ===")
        backbone = build_backbone(name, class_names=class_names if name == "clip_vit_b_32" else None)
        backbone.to(args.device)

        head = LinearHead(backbone.feature_dim, len(class_names))
        head.load_state_dict(load_head_state(results_dir, name))
        head.to(args.device).eval()

        clean = load_predictions(results_dir, f"clean_{name}")
        metrics, preds, probs, labels = evaluate_and_compare(
            backbone, head, shuffled_ds, args.device, clean["preds"]
        )
        print(f"  [{name} head] {metrics}")
        save_predictions(results_dir, f"patch_shuffle_{name}", preds, probs, labels)
        results["models"][name] = {"head": metrics}

        if name == "clip_vit_b_32":
            clean_zs = load_predictions(results_dir, "clean_clip_zero_shot")
            metrics_zs, preds_zs, probs_zs, labels_zs = evaluate_and_compare(
                backbone, None, shuffled_ds, args.device, clean_zs["preds"], is_zero_shot=True
            )
            print(f"  [clip zero-shot] {metrics_zs}")
            save_predictions(results_dir, "patch_shuffle_clip_zero_shot", preds_zs, probs_zs, labels_zs)
            results["models"]["clip_zero_shot"] = metrics_zs

    out_path = Path(results_dir) / "patch_shuffle_seed6304.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[patch_shuffle] wrote {out_path}")


if __name__ == "__main__":
    main()
