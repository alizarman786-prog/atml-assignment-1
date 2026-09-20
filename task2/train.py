"""Task 2 training entry point. Currently supports method=source_only and
method=dan; dann/cdan will plug into the same splits/iterators/evaluation
once added.

Usage (from inside task2/):
    python train.py --config configs/source_only.yaml
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
sys.path.append(str(Path(__file__).resolve().parents[1] / "shared"))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backbone import build_pacs_model  # noqa: E402
from pacs import load_pacs_hf, PACSSubset, PACS_TRAIN_TRANSFORM, PACS_EVAL_TRANSFORM  # noqa: E402
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

    print("[task2] loading PACS (downloads on first use, cached after)...")
    hf_dataset = load_pacs_hf(cache_dir=cfg["data_root"])

    splits_path = Path(cfg["splits_path"])
    if splits_path.exists():
        print(f"[task2] loading existing splits from {splits_path}")
        splits = load_pacs_splits(str(splits_path))
    else:
        print(f"[task2] building splits (seed {cfg['seed']}) -> {splits_path}")
        splits = build_pacs_splits(hf_dataset, out_path=str(splits_path))

    source_data = build_source_datasets(hf_dataset, splits)
    for domain, d in source_data.items():
        print(f"  {domain}: train={len(d['train'])} val={len(d['val'])}")

    model = build_pacs_model(num_classes=cfg["num_classes"])

    train_datasets = {d: source_data[d]["train"] for d in cfg["source_domains"]}
    val_datasets = {d: source_data[d]["val"] for d in cfg["source_domains"]}

    hp = cfg["training"]
    train_iterators = build_domain_iterators(
        train_datasets, batch_size_per_domain=hp["batch_size_per_domain"],
        transform=PACS_TRAIN_TRANSFORM,
    )

    if cfg["method"] == "source_only":
        from source_only import train_source_only
        print("[task2] training Source-only ERM...")
        model, history = train_source_only(
            model, train_iterators, val_datasets, args.device, PACS_EVAL_TRANSFORM,
            max_epochs=hp["max_epochs"], steps_per_epoch=hp["steps_per_epoch"],
            lr=hp["lr"], weight_decay=hp["weight_decay"],
            patience=hp["early_stopping_patience"],
        )
    elif cfg["method"] == "dan":
        from dan import train_dan
        target_ds = PACSSubset(hf_dataset, splits["target_idx"])
        target_iterator = build_domain_iterators(
            {"sketch": target_ds}, batch_size_per_domain=hp["target_batch_size"],
            transform=PACS_TRAIN_TRANSFORM,
        )["sketch"]
        print(f"[task2] training DAN (lambda_mmd={hp['lambda_mmd']})...")
        model, history = train_dan(
            model, train_iterators, target_iterator, val_datasets, args.device, PACS_EVAL_TRANSFORM,
            lambda_mmd=hp["lambda_mmd"], max_epochs=hp["max_epochs"], steps_per_epoch=hp["steps_per_epoch"],
            lr=hp["lr"], weight_decay=hp["weight_decay"],
            patience=hp["early_stopping_patience"],
        )
    else:
        raise NotImplementedError(f"method {cfg['method']!r} not yet implemented")

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['method']}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[task2] saved checkpoint -> {ckpt_path}")

    out_path = Path(cfg["output"]["results_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "method": cfg["method"],
        "seed": cfg["seed"],
        "history": history,
    }, indent=2))
    print(f"[task2] wrote {out_path}")


if __name__ == "__main__":
    main()
