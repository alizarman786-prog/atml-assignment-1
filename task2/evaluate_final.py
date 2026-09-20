"""Task 2, Step 5: Common Evaluation and Alignment Diagnostic.

Loads whichever of {source_only, dan, dann, cdan} checkpoints already exist
(skips missing ones with a warning, so this can be run incrementally as
methods finish training), and for each: source-validation per-domain/mean
accuracy+macro-F1, target accuracy+macro-F1 (+ change vs. Source-only),
domain separability, and per-class target accuracy vs. the Source-only
baseline (most improved / most degraded classes).

Usage (from inside task2/, after training whichever methods you want compared):
    python evaluate_final.py --config configs/source_only.yaml --device cuda
(any one method's config works -- they all point at the same splits/data_root)
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
from pacs import load_pacs_hf, PACSSubset, PACS_EVAL_TRANSFORM  # noqa: E402
from pacs_protocol import load_pacs_splits  # noqa: E402
from metrics import evaluate_domain, evaluate_source_domains  # noqa: E402
from domain_separability import domain_separability  # noqa: E402
from class_analysis import compare_to_baseline  # noqa: E402
from common.metrics import top1_accuracy, macro_f1  # noqa: E402

METHODS = ["source_only", "dan", "dann", "cdan"]
BASELINE = "source_only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="any one method's config (shares data_root/splits_path)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/task2_final_comparison.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    hf_dataset = load_pacs_hf(cache_dir=cfg["data_root"])
    splits = load_pacs_splits(cfg["splits_path"])
    class_names = hf_dataset.features["label"].names

    target_ds = PACSSubset(hf_dataset, splits["target_idx"])
    source_val_datasets = {
        domain: PACSSubset(hf_dataset, idx["val_idx"])
        for domain, idx in splits["source_domains"].items()
    }

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    results = {"methods": {}}
    target_preds_by_method = {}
    target_labels = None  # same for every method (same target_ds, deterministic order)

    for method in METHODS:
        ckpt_path = ckpt_dir / f"{method}.pt"
        if not ckpt_path.exists():
            print(f"[evaluate_final] skipping {method!r}: no checkpoint at {ckpt_path}")
            continue

        print(f"\n[evaluate_final] === {method} ===")
        model = build_pacs_model(num_classes=cfg["num_classes"]).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        source_eval = evaluate_source_domains(model, source_val_datasets, args.device, PACS_EVAL_TRANSFORM)
        preds, labels = evaluate_domain(model, target_ds, args.device, PACS_EVAL_TRANSFORM)
        target_labels = labels
        target_metrics = {"accuracy": top1_accuracy(preds, labels), "macro_f1": macro_f1(preds, labels)}
        target_preds_by_method[method] = preds

        sep = domain_separability(model, source_val_datasets, target_ds, args.device, PACS_EVAL_TRANSFORM)

        print(f"  source mean macro-F1: {source_eval['mean_macro_f1']:.4f}  "
              f"target: {target_metrics}  domain_sep_acc: {sep['domain_separability_accuracy']:.4f}")

        results["methods"][method] = {
            "source_per_domain": source_eval["per_domain"],
            "source_mean_accuracy": source_eval["mean_accuracy"],
            "source_mean_macro_f1": source_eval["mean_macro_f1"],
            "target_metrics": target_metrics,
            "domain_separability": sep,
        }

    if BASELINE in results["methods"]:
        baseline_acc = results["methods"][BASELINE]["target_metrics"]["accuracy"]
        baseline_preds = target_preds_by_method[BASELINE]

        for method, m_results in results["methods"].items():
            m_results["target_accuracy_change_vs_baseline"] = (
                m_results["target_metrics"]["accuracy"] - baseline_acc
            )
            if method != BASELINE:
                m_results["per_class_vs_baseline"] = compare_to_baseline(
                    baseline_preds, target_preds_by_method[method], target_labels, class_names,
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[evaluate_final] wrote {out_path}")


if __name__ == "__main__":
    main()
