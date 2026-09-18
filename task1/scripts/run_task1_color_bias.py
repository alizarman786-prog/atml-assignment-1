"""Task 1, Step 2: Color Bias.

Applies grayscale (required) and one additional color transform (fixed hue
rotation, implemented here) to the eval subset, reruns each trained head +
CLIP zero-shot, and reports accuracy change and prediction consistency
relative to the clean-baseline predictions saved in Step 1.

Requires Step 1 (run_task1_clean_baseline.py) to have been run first, since
it reuses those saved head checkpoints and clean predictions rather than
retraining or recomputing them.

Usage:
    python scripts/run_task1_color_bias.py --config configs/clean_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[3]))

from dataset import build_split_datasets  # noqa: E402
from transformed_dataset import TransformedDataset  # noqa: E402
from transforms import grayscale, hue_rotate  # noqa: E402
from backbones import build_backbone, LinearHead  # noqa: E402
from extract_features import extract_features  # noqa: E402
from checkpoints import load_head_state, load_predictions, save_predictions  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.metrics import (  # noqa: E402
    top1_accuracy, macro_f1, mean_max_confidence, prediction_consistency,
)

COLOR_INTERVENTIONS = {
    "grayscale": grayscale,
    "hue_rotate_90": partial(hue_rotate, degrees=90.0),
}


def evaluate_and_compare(preds, probs, labels, clean_preds):
    return {
        "top1_accuracy": top1_accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels),
        "mean_max_confidence": mean_max_confidence(probs),
        "accuracy_delta_vs_clean": top1_accuracy(preds, labels) - top1_accuracy(clean_preds, labels),
        "prediction_consistency_vs_clean": prediction_consistency(preds, clean_preds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    results_dir = str(Path(cfg["output"]["results_json"]).parent)

    print(f"[color_bias] dataset={cfg['dataset']} device={args.device}")
    _, _, eval_ds = build_split_datasets(cfg["dataset"], cfg["data_root"], cfg["splits_path"])
    class_names = eval_ds.class_names

    results = {"dataset": cfg["dataset"], "seed": cfg["seed"], "interventions": {}}

    for intervention_name, transform_fn in COLOR_INTERVENTIONS.items():
        print(f"\n[color_bias] === {intervention_name} ===")
        transformed_eval_ds = TransformedDataset(eval_ds, transform_fn)
        results["interventions"][intervention_name] = {}

        for name in cfg["backbones"]:
            backbone = build_backbone(name, class_names=class_names if name == "clip_vit_b_32" else None)
            backbone.to(args.device)

            eval_feats, eval_labels = extract_features(backbone, transformed_eval_ds, device=args.device)
            labels_np = eval_labels.numpy()

            # --- trained linear head ---
            head = LinearHead(backbone.feature_dim, len(class_names))
            head.load_state_dict(load_head_state(results_dir, name))
            head.to(args.device).eval()
            with torch.no_grad():
                logits = head(eval_feats.to(args.device)).cpu()
                probs = F.softmax(logits, dim=1).numpy()
                preds = probs.argmax(axis=1)

            clean = load_predictions(results_dir, f"clean_{name}")
            metrics = evaluate_and_compare(preds, probs, labels_np, clean["preds"])
            print(f"  [{name} head] {metrics}")
            save_predictions(results_dir, f"{intervention_name}_{name}", preds, probs, labels_np)
            results["interventions"][intervention_name][name] = {"head": metrics}

            # --- CLIP zero-shot ---
            if name == "clip_vit_b_32":
                loader = DataLoader(
                    transformed_eval_ds, batch_size=64, shuffle=False,
                    collate_fn=lambda batch: (
                        torch.stack([backbone.preprocess(img) for img, _ in batch]),
                        torch.tensor([lab for _, lab in batch]),
                    ),
                )
                all_probs, all_labels = [], []
                with torch.no_grad():
                    for imgs, labs in loader:
                        p = backbone.zero_shot_predict(imgs.to(args.device)).cpu()
                        all_probs.append(p)
                        all_labels.append(labs)
                probs_zs = torch.cat(all_probs).numpy()
                labels_zs = torch.cat(all_labels).numpy()
                preds_zs = probs_zs.argmax(axis=1)

                clean_zs = load_predictions(results_dir, "clean_clip_zero_shot")
                zs_metrics = evaluate_and_compare(preds_zs, probs_zs, labels_zs, clean_zs["preds"])
                print(f"  [clip zero-shot] {zs_metrics}")
                save_predictions(results_dir, f"{intervention_name}_clip_zero_shot", preds_zs, probs_zs, labels_zs)
                results["interventions"][intervention_name]["clip_zero_shot"] = zs_metrics

    out_path = Path(results_dir) / "color_bias_seed6304.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[color_bias] wrote {out_path}")


if __name__ == "__main__":
    main()