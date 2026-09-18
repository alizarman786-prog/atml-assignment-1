"""Task 1, Step 4: Translation.

For displacements 0/8/16/32 pixels in each of the 4 cardinal directions
(reflection-padded shift), reruns each trained head + CLIP zero-shot and
reports accuracy and prediction consistency relative to clean, per the
assignment's Consistency(delta) formula. The per-direction results at each
displacement are averaged together (as the assignment specifies) into a
single accuracy/consistency curve point per delta.

Requires run_task1_clean_baseline.py to have been run first (reuses the
saved heads and clean predictions).

Usage (from inside task1/):
    python scripts/run_task1_translation.py --config configs/clean_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset import build_split_datasets  # noqa: E402
from transformed_dataset import TransformedDataset  # noqa: E402
from transforms import CARDINAL_DIRECTIONS, translate_reflect  # noqa: E402
from backbones import build_backbone, LinearHead  # noqa: E402
from checkpoints import load_head_state, load_predictions  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.metrics import top1_accuracy, prediction_consistency  # noqa: E402

DISPLACEMENTS = [0, 8, 16, 32]


def make_translate_fn(direction: str, delta: int):
    dx_sign, dy_sign = CARDINAL_DIRECTIONS[direction]

    def _fn(img):
        return translate_reflect(img, dx=dx_sign * delta, dy=dy_sign * delta)

    return _fn


def evaluate_one(backbone, head_or_none, ds, device, is_zero_shot=False):
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
    return preds, labels


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

    results = {"dataset": cfg["dataset"], "seed": cfg["seed"], "displacements": DISPLACEMENTS, "models": {}}

    for name in cfg["backbones"]:
        print(f"\n[translation] === {name} ===")
        backbone = build_backbone(name, class_names=class_names if name == "clip_vit_b_32" else None)
        backbone.to(args.device)

        head = LinearHead(backbone.feature_dim, len(class_names))
        head.load_state_dict(load_head_state(results_dir, name))
        head.to(args.device).eval()

        clean = load_predictions(results_dir, f"clean_{name}")
        by_delta = {}
        for delta in DISPLACEMENTS:
            dir_accs, dir_cons = [], []
            for direction in CARDINAL_DIRECTIONS:
                ds = TransformedDataset(eval_ds, make_translate_fn(direction, delta))
                preds, labels = evaluate_one(backbone, head, ds, args.device)
                dir_accs.append(top1_accuracy(preds, labels))
                dir_cons.append(prediction_consistency(preds, clean["preds"]))
            by_delta[str(delta)] = {
                "accuracy": float(np.mean(dir_accs)),
                "consistency": float(np.mean(dir_cons)),
            }
            print(f"  delta={delta}px: accuracy={by_delta[str(delta)]['accuracy']:.4f} "
                  f"consistency={by_delta[str(delta)]['consistency']:.4f}")
        results["models"][name] = {"head_by_delta": by_delta}

        if name == "clip_vit_b_32":
            clean_zs = load_predictions(results_dir, "clean_clip_zero_shot")
            by_delta_zs = {}
            for delta in DISPLACEMENTS:
                dir_accs, dir_cons = [], []
                for direction in CARDINAL_DIRECTIONS:
                    ds = TransformedDataset(eval_ds, make_translate_fn(direction, delta))
                    preds, labels = evaluate_one(backbone, None, ds, args.device, is_zero_shot=True)
                    dir_accs.append(top1_accuracy(preds, labels))
                    dir_cons.append(prediction_consistency(preds, clean_zs["preds"]))
                by_delta_zs[str(delta)] = {
                    "accuracy": float(np.mean(dir_accs)),
                    "consistency": float(np.mean(dir_cons)),
                }
                print(f"  [zero-shot] delta={delta}px: accuracy={by_delta_zs[str(delta)]['accuracy']:.4f} "
                      f"consistency={by_delta_zs[str(delta)]['consistency']:.4f}")
            results["models"]["clip_zero_shot"] = {"by_delta": by_delta_zs}

    out_path = Path(results_dir) / "translation_seed6304.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[translation] wrote {out_path}")


if __name__ == "__main__":
    main()
