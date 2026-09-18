"""Task 1, Step 1: Clean Baseline.

For each of ResNet-50, ViT-B/16, CLIP ViT-B-32:
  1. Extract frozen features for train/val (official train partition split)
     and the eval subset (official test partition, class-balanced 500 imgs).
  2. Train a LinearHead on train features, early-stopped on val accuracy.
  3. Evaluate the trained head on the eval subset: top-1 accuracy, macro-F1,
     mean max confidence.
Also evaluates CLIP zero-shot on the eval subset with the fixed prompt
template, using the same three metrics (confidence from the scaled-similarity
softmax).

Usage:
    python scripts/run_task1_clean_baseline.py --config configs/clean_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset import build_split_datasets  # noqa: E402
from backbones import build_backbone, LinearHead  # noqa: E402
from extract_features import extract_features  # noqa: E402
from train_head import train_linear_head  # noqa: E402
from checkpoints import save_head, save_predictions  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.metrics import top1_accuracy, macro_f1, mean_max_confidence  # noqa: E402


def evaluate_head(head, feats, labels, device):
    head.eval()
    with torch.no_grad():
        logits = head(feats.to(device)).cpu()
        probs = F.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
    metrics = {
        "top1_accuracy": top1_accuracy(preds, labels.numpy()),
        "macro_f1": macro_f1(preds, labels.numpy()),
        "mean_max_confidence": mean_max_confidence(probs),
    }
    return metrics, preds, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    print(f"[run_task1] dataset={cfg['dataset']} device={args.device}")
    train_ds, val_ds, eval_ds = build_split_datasets(
        cfg["dataset"], cfg["data_root"], cfg["splits_path"]
    )
    class_names = train_ds.class_names
    print(f"[run_task1] train={len(train_ds)} val={len(val_ds)} eval={len(eval_ds)} "
          f"classes={len(class_names)}")

    results = {"dataset": cfg["dataset"], "seed": cfg["seed"], "backbones": {}}

    for name in cfg["backbones"]:
        print(f"\n[run_task1] === {name} ===")
        backbone = build_backbone(name, class_names=class_names if name == "clip_vit_b_32" else None)

        print("  extracting features (train/val/eval)...")
        train_feats, train_labels = extract_features(backbone, train_ds, device=args.device)
        val_feats, val_labels = extract_features(backbone, val_ds, device=args.device)
        eval_feats, eval_labels = extract_features(backbone, eval_ds, device=args.device)

        print("  training linear head...")
        head = LinearHead(backbone.feature_dim, len(class_names))
        hp = cfg["head_training"]
        head, history = train_linear_head(
            head, train_feats, train_labels, val_feats, val_labels,
            max_epochs=hp["max_epochs"], lr=hp["lr"], weight_decay=hp["weight_decay"],
            patience=hp["early_stopping_patience"], device=args.device,
        )

        head_metrics, head_preds, head_probs = evaluate_head(head, eval_feats, eval_labels, args.device)
        print(f"  [{name} head] eval: {head_metrics}")

        results_dir = str(Path(cfg["output"]["results_json"]).parent)
        ckpt_path = save_head(head, results_dir, name)
        save_predictions(results_dir, f"clean_{name}", head_preds, head_probs, eval_labels.numpy())
        print(f"  saved head checkpoint -> {ckpt_path}")

        backbone_result = {
            "head_val_best_acc": history["best_val_acc"],
            "head_epochs_trained": history["epochs_trained"],
            "clean_eval_head": head_metrics,
        }

        if name == "clip_vit_b_32":
            print("  evaluating CLIP zero-shot...")
            backbone.to(args.device)
            from torch.utils.data import DataLoader

            loader = DataLoader(
                eval_ds, batch_size=64, shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([backbone.preprocess(img) for img, _ in batch]),
                    torch.tensor([lab for _, lab in batch]),
                ),
            )
            all_probs, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in loader:
                    probs = backbone.zero_shot_predict(imgs.to(args.device)).cpu()
                    all_probs.append(probs)
                    all_labels.append(labels)
            probs = torch.cat(all_probs).numpy()
            labels = torch.cat(all_labels).numpy()
            preds = probs.argmax(axis=1)
            zs_metrics = {
                "top1_accuracy": top1_accuracy(preds, labels),
                "macro_f1": macro_f1(preds, labels),
                "mean_max_confidence": mean_max_confidence(probs),
            }
            print(f"  [clip zero-shot] eval: {zs_metrics}")
            backbone_result["clean_eval_zero_shot"] = zs_metrics
            save_predictions(results_dir, "clean_clip_zero_shot", preds, probs, labels)

        results["backbones"][name] = backbone_result

    out_path = Path(cfg["output"]["results_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[run_task1] wrote {out_path}")


if __name__ == "__main__":
    main()