"""Task 3, Step 4: Common Evaluation and Diagnostics.

Loads whichever of {erm, dan_dg, sam} checkpoints exist, and for each:
source-validation per-domain/mean/worst accuracy+macro-F1, Sketch
accuracy+macro-F1 (+ change vs. ERM), source-domain separability (3-way),
sharpness proxy, and per-class Sketch changes vs. ERM. Also loads Task 2's
DAN and Source-only target results (if present) for the required
Task2-vs-Task3 comparison.

Sketch labels are used only here, after all training/selection decisions
for every method are fixed.

Usage (from inside task3/, after training whichever methods you want compared):
    python evaluate_sketch.py --device cuda
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

sys.path.append(str(Path(__file__).resolve().parents[2] / "task2" / "evaluation"))
from class_analysis import compare_to_baseline  # noqa: E402

METHODS = ["erm", "dan_dg", "sam"]
BASELINE = "erm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/erm.yaml", help="any one method's config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/task3_final_comparison.json")
    ap.add_argument("--task2_final", default="../task2/results/task2_final_comparison.json",
                     help="Task 2's combined comparison, for the required cross-task comparison")
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
    source_train_datasets = {
        domain: PACSSubset(hf_dataset, idx["train_idx"])
        for domain, idx in splits["source_domains"].items()
    }

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    results = {"methods": {}}
    target_preds_by_method = {}
    target_labels = None

    for method in METHODS:
        ckpt_path = ckpt_dir / f"{method}.pt"
        if not ckpt_path.exists():
            print(f"[evaluate_sketch] skipping {method!r}: no checkpoint at {ckpt_path}")
            continue

        print(f"\n[evaluate_sketch] === {method} ===")
        model = build_pacs_model(num_classes=cfg["num_classes"]).to(args.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        source_eval = evaluate_source_domains(model, source_val_datasets, args.device, PACS_EVAL_TRANSFORM)
        preds, labels = evaluate_domain(model, target_ds, args.device, PACS_EVAL_TRANSFORM)
        target_labels = labels
        target_metrics = {"accuracy": top1_accuracy(preds, labels), "macro_f1": macro_f1(preds, labels)}
        target_preds_by_method[method] = preds

        sep = source_domain_separability(model, source_val_datasets, args.device, PACS_EVAL_TRANSFORM)
        sharp = sharpness_proxy(model, source_train_datasets, PACS_TRAIN_TRANSFORM, args.device, rho=0.05)

        print(f"  source mean_macro_f1={source_eval['mean_macro_f1']:.4f} "
              f"worst_macro_f1={source_eval['worst_macro_f1']:.4f}  "
              f"target={target_metrics}  "
              f"source_sep_acc={sep['source_domain_separability_accuracy']:.4f}  "
              f"delta_sharp={sharp['delta_sharp']:.4f}")

        results["methods"][method] = {
            "source_per_domain": source_eval["per_domain"],
            "source_mean_accuracy": source_eval["mean_accuracy"],
            "source_mean_macro_f1": source_eval["mean_macro_f1"],
            "source_worst_accuracy": source_eval["worst_accuracy"],
            "source_worst_macro_f1": source_eval["worst_macro_f1"],
            "target_metrics": target_metrics,
            "source_domain_separability": sep,
            "sharpness": sharp,
        }

    if BASELINE in results["methods"]:
        baseline_acc = results["methods"][BASELINE]["target_metrics"]["accuracy"]
        baseline_preds = target_preds_by_method[BASELINE]
        for method, m_results in results["methods"].items():
            m_results["target_accuracy_change_vs_erm"] = (
                m_results["target_metrics"]["accuracy"] - baseline_acc
            )
            if method != BASELINE:
                m_results["per_class_vs_erm"] = compare_to_baseline(
                    baseline_preds, target_preds_by_method[method], target_labels, class_names,
                )

    # Required comparison: Task 2's target-aware DAN vs. Task 3's target-free DAN-DG,
    # both relative to the SAME shared ERM/Source-only baseline.
    task2_path = Path(args.task2_final)
    if task2_path.exists() and "dan_dg" in results["methods"]:
        task2_results = json.loads(task2_path.read_text())
        task2_methods = task2_results.get("methods", {})
        if "source_only" in task2_methods and "dan" in task2_methods:
            results["task2_vs_task3_dan_comparison"] = {
                "task2_source_only_target_accuracy": task2_methods["source_only"]["target_metrics"]["accuracy"],
                "task2_dan_target_accuracy": task2_methods["dan"]["target_metrics"]["accuracy"],
                "task2_dan_change_vs_source_only": task2_methods["dan"]["target_metrics"]["accuracy"]
                    - task2_methods["source_only"]["target_metrics"]["accuracy"],
                "task3_erm_target_accuracy": results["methods"]["erm"]["target_metrics"]["accuracy"],
                "task3_dan_dg_target_accuracy": results["methods"]["dan_dg"]["target_metrics"]["accuracy"],
                "task3_dan_dg_change_vs_erm": results["methods"]["dan_dg"]["target_accuracy_change_vs_erm"],
                "note": "Task2's DAN sees unlabeled Sketch during training; Task3's DAN-DG never does. "
                        "Both changes are measured relative to the SAME underlying Source-only/ERM checkpoint.",
            }
            print("\n[evaluate_sketch] Task 2 DAN vs Task 3 DAN-DG comparison added.")
        else:
            print("\n[evaluate_sketch] Task 2 results found but missing source_only/dan methods, skipping cross-task comparison")
    else:
        print(f"\n[evaluate_sketch] Task 2 final comparison not found at {task2_path}, skipping cross-task comparison")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[evaluate_sketch] wrote {out_path}")


if __name__ == "__main__":
    main()
