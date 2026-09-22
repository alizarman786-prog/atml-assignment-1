"""Task 3 training entry point. Supports method in {erm, dan_dg, sam}.
ERM is not actually "trained" here -- it loads and copies Task 2's
Source-only checkpoint unchanged, per the assignment.

Usage (from inside task3/):
    python train.py --config configs/erm.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent / "models"))
sys.path.append(str(Path(__file__).resolve().parent / "methods"))
sys.path.append(str(Path(__file__).resolve().parent / "evaluation"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backbone import build_pacs_model  # noqa: E402
from pacs import load_pacs_hf, PACS_TRAIN_TRANSFORM, PACS_EVAL_TRANSFORM  # noqa: E402
from pacs_protocol import (  # noqa: E402
    build_pacs_splits, load_pacs_splits, build_source_datasets, build_domain_iterators,
)
from common.seed import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    print("[task3] loading PACS (downloads on first use, cached after)...")
    hf_dataset = load_pacs_hf(cache_dir=cfg["data_root"])

    splits_path = Path(cfg["splits_path"])
    if splits_path.exists():
        print(f"[task3] loading existing splits from {splits_path}")
        splits = load_pacs_splits(str(splits_path))
    else:
        print(f"[task3] building splits (seed {cfg['seed']}) -> {splits_path}")
        splits = build_pacs_splits(hf_dataset, out_path=str(splits_path))

    source_data = build_source_datasets(hf_dataset, splits)
    for domain, d in source_data.items():
        print(f"  {domain}: train={len(d['train'])} val={len(d['val'])}")

    train_datasets = {d: source_data[d]["train"] for d in cfg["source_domains"]}
    val_datasets = {d: source_data[d]["val"] for d in cfg["source_domains"]}

    model = build_pacs_model(num_classes=cfg["num_classes"])
    hp = cfg["training"]

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['method']}.pt"

    if cfg["method"] == "erm":
        from erm import load_erm_from_task2, copy_checkpoint
        from domain_metrics import evaluate_source_domains
        print(f"[task3] loading ERM baseline from Task 2 checkpoint: {cfg['task2_checkpoint']}")
        model = load_erm_from_task2(model, cfg["task2_checkpoint"], args.device)
        copy_checkpoint(cfg["task2_checkpoint"], str(ckpt_path))
        eval_result = evaluate_source_domains(model, val_datasets, args.device, PACS_EVAL_TRANSFORM)
        history = {
            "note": "ERM reuses Task 2's Source-only checkpoint unchanged; not retrained.",
            "val_mean_macro_f1": eval_result["mean_macro_f1"],
            "val_per_domain": eval_result["per_domain"],
        }
        print(f"  ERM source val: mean_macro_f1={eval_result['mean_macro_f1']:.4f} "
              f"per_domain={ {d: round(m['macro_f1'], 3) for d, m in eval_result['per_domain'].items()} }")

    elif cfg["method"] == "dan_dg":
        from dan_dg import train_dan_dg
        train_iterators = build_domain_iterators(
            train_datasets, batch_size_per_domain=hp["batch_size_per_domain"],
            transform=PACS_TRAIN_TRANSFORM,
        )
        print(f"[task3] training DAN-DG (lambda_dg={hp['lambda_dg']})...")
        model, history = train_dan_dg(
            model, train_iterators, val_datasets, args.device, PACS_EVAL_TRANSFORM,
            lambda_dg=hp["lambda_dg"], max_epochs=hp["max_epochs"], steps_per_epoch=hp["steps_per_epoch"],
            lr=hp["lr"], weight_decay=hp["weight_decay"], patience=hp["early_stopping_patience"],
        )
        torch.save(model.state_dict(), ckpt_path)

    elif cfg["method"] == "sam":
        from sam import train_sam
        train_iterators = build_domain_iterators(
            train_datasets, batch_size_per_domain=hp["batch_size_per_domain"],
            transform=PACS_TRAIN_TRANSFORM,
        )
        print(f"[task3] training SAM (rho={hp['rho']})...")
        model, history = train_sam(
            model, train_iterators, val_datasets, args.device, PACS_EVAL_TRANSFORM,
            rho=hp["rho"], max_epochs=hp["max_epochs"], steps_per_epoch=hp["steps_per_epoch"],
            lr=hp["lr"], weight_decay=hp["weight_decay"], patience=hp["early_stopping_patience"],
        )
        torch.save(model.state_dict(), ckpt_path)

    else:
        raise NotImplementedError(f"method {cfg['method']!r} not yet implemented")

    print(f"[task3] saved checkpoint -> {ckpt_path}")

    out_path = Path(cfg["output"]["results_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "method": cfg["method"],
        "seed": cfg["seed"],
        "history": history,
    }, indent=2))
    print(f"[task3] wrote {out_path}")


if __name__ == "__main__":
    main()
