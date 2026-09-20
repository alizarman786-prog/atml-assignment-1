"""Evaluates a trained Task 2 checkpoint on the full Sketch target domain.
Per the assignment, target labels are used only at this final-analysis
stage, after all models/settings/checkpoints have been fixed -- run this
only once you are done tuning the corresponding method.

Usage (from inside task2/):
    python evaluate_target.py --config configs/source_only.yaml --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent / "models"))
sys.path.append(str(Path(__file__).resolve().parent / "evaluation"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))

from backbone import build_pacs_model  # noqa: E402
from pacs import load_pacs_hf, PACSSubset, PACS_EVAL_TRANSFORM  # noqa: E402
from pacs_protocol import load_pacs_splits  # noqa: E402
from metrics import evaluate_domain  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.metrics import top1_accuracy, macro_f1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    hf_dataset = load_pacs_hf(cache_dir=cfg["data_root"])
    splits = load_pacs_splits(cfg["splits_path"])
    target_ds = PACSSubset(hf_dataset, splits["target_idx"])
    print(f"[evaluate_target] target domain ({cfg['target_domain']}) size: {len(target_ds)}")

    model = build_pacs_model(num_classes=cfg["num_classes"]).to(args.device)
    ckpt_path = Path(cfg["output"]["checkpoints_dir"]) / f"{cfg['method']}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    print(f"[evaluate_target] loaded checkpoint {ckpt_path}")

    preds, labels = evaluate_domain(model, target_ds, args.device, PACS_EVAL_TRANSFORM)
    target_metrics = {
        "accuracy": top1_accuracy(preds, labels),
        "macro_f1": macro_f1(preds, labels),
    }
    print(f"[evaluate_target] {cfg['method']} on {cfg['target_domain']}: {target_metrics}")

    results_path = Path(cfg["output"]["results_json"])
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["target_domain"] = cfg["target_domain"]
    results["target_metrics"] = target_metrics
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[evaluate_target] updated {results_path}")


if __name__ == "__main__":
    main()
