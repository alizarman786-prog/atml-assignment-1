"""Task 3, Step 5: summarizes DAN-DG's lambda_DG controlled study
(lambda_dg in {0.1, 1, 10}) -- source performance, source-domain
separability, sharpness proxy, and Sketch performance for each setting.

Usage (from inside task3/, after training all three lambda_dg settings):
    python evaluate_controlled_study.py --device cuda
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
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backbone import build_pacs_model  # noqa: E402
from pacs import load_pacs_hf, PACSSubset, PACS_TRAIN_TRANSFORM, PACS_EVAL_TRANSFORM  # noqa: E402
from pacs_protocol import load_pacs_splits  # noqa: E402
from domain_metrics import evaluate_domain, evaluate_source_domains  # noqa: E402
from source_domain_separability import source_domain_separability  # noqa: E402
from sharpness import sharpness_proxy  # noqa: E402
from common.metrics import top1_accuracy, macro_f1  # noqa: E402

SETTINGS = [
    ("configs/dan_dg_lambda01.yaml", "results/checkpoints_lambda01", "lambda_dg=0.1"),
    ("configs/dan_dg.yaml", "results/checkpoints", "lambda_dg=1.0 (main)"),
    ("configs/dan_dg_lambda10.yaml", "results/checkpoints_lambda10", "lambda_dg=10.0"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/task3_controlled_study.json")
    args = ap.parse_args()

    results = {"study": "DAN-DG lambda_DG", "settings": {}}

    for config_path, ckpt_dir, label in SETTINGS:
        cfg = yaml.safe_load(Path(config_path).read_text())
        ckpt_path = Path(ckpt_dir) / "dan_dg.pt"
        if not ckpt_path.exists():
            print(f"[controlled_study] skipping {label}: no checkpoint at {ckpt_path}")
            continue

        print(f"\n[controlled_study] === {label} ===")
        hf_dataset = load_pacs_hf(cache_dir=cfg["data_root"])
        splits = load_pacs_splits(cfg["splits_path"])
        target_ds = PACSSubset(hf_dataset, splits["target_idx"])
        source_val_datasets = {
            domain: PACSSubset(hf_dataset, idx["val_idx"])
            for domain, idx in splits["source_domains"].items()
        }
        source_train_datasets = {
            domain: PACSSubset(hf_dataset, idx["train_idx"])
            for domain, idx in splits["source_domains"].items()
        }

        model = build_pacs_model(num_classes=cfg["num_classes"]).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        source_eval = evaluate_source_domains(model, source_val_datasets, args.device, PACS_EVAL_TRANSFORM)
        preds, labels = evaluate_domain(model, target_ds, args.device, PACS_EVAL_TRANSFORM)
        target_metrics = {"accuracy": top1_accuracy(preds, labels), "macro_f1": macro_f1(preds, labels)}
        sep = source_domain_separability(model, source_val_datasets, args.device, PACS_EVAL_TRANSFORM)
        sharp = sharpness_proxy(model, source_train_datasets, PACS_TRAIN_TRANSFORM, args.device, rho=0.05)

        print(f"  source_mean_macro_f1={source_eval['mean_macro_f1']:.4f}  "
              f"source_sep_acc={sep['source_domain_separability_accuracy']:.4f}  "
              f"delta_sharp={sharp['delta_sharp']:.4f}  target={target_metrics}")

        results["settings"][label] = {
            "source_mean_accuracy": source_eval["mean_accuracy"],
            "source_mean_macro_f1": source_eval["mean_macro_f1"],
            "source_domain_separability_accuracy": sep["source_domain_separability_accuracy"],
            "delta_sharp": sharp["delta_sharp"],
            "target_accuracy": target_metrics["accuracy"],
            "target_macro_f1": target_metrics["macro_f1"],
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[controlled_study] wrote {out_path}")


if __name__ == "__main__":
    main()
