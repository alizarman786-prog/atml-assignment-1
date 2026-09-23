"""Task 4 training entry point. Supports method in {vanilla, gcsc, proser}.

Usage (from inside task4/):
    python train.py --config configs/vanilla.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent / "data"))
sys.path.append(str(Path(__file__).resolve().parent / "models"))
sys.path.append(str(Path(__file__).resolve().parent / "methods"))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from cifar10 import load_cifar10_splits, TRAIN_TRANSFORM, EVAL_TRANSFORM  # noqa: E402
from resnet_cifar import build_cifar_resnet18  # noqa: E402
from common.seed import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg['method']}.pt"

    hp = cfg["training"]

    if cfg["method"] == "vanilla":
        from vanilla import train_vanilla
        train_ds, val_ds, _test_ds = load_cifar10_splits(cfg["data_root"])
        model = build_cifar_resnet18(num_classes=cfg["num_classes"])
        print("[task4] training Vanilla...")
        model, history = train_vanilla(
            model, train_ds, val_ds, args.device,
            max_epochs=hp["max_epochs"], batch_size=hp["batch_size"],
            lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["weight_decay"],
            seed=cfg["seed"],
        )
        torch.save(model.state_dict(), ckpt_path)

    elif cfg["method"] == "gcsc":
        from gcsc import GCSC_TRAIN_TRANSFORM, train_gcsc
        train_ds, val_ds, _test_ds = load_cifar10_splits(
            cfg["data_root"], train_transform=GCSC_TRAIN_TRANSFORM,
        )
        model = build_cifar_resnet18(num_classes=cfg["num_classes"])
        print("[task4] training GCSC...")
        model, history = train_gcsc(
            model, train_ds, val_ds, args.device,
            max_epochs=hp["max_epochs"], batch_size=hp["batch_size"],
            lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["weight_decay"],
            seed=cfg["seed"],
        )
        torch.save(model.state_dict(), ckpt_path)

    elif cfg["method"] == "proser":
        from proser import train_proser
        train_ds, val_ds, _test_ds = load_cifar10_splits(cfg["data_root"])
        print(f"[task4] training PROSER from {cfg['vanilla_checkpoint']}...")
        model, history = train_proser(
            cfg["vanilla_checkpoint"], train_ds, val_ds, args.device,
            num_known=cfg["num_classes"], num_dummy=hp["num_dummy"],
            beta=hp["beta"], gamma=hp["gamma"],
            max_epochs=hp["max_epochs"], batch_size=hp["batch_size"],
            lr=hp["lr"], momentum=hp["momentum"], weight_decay=hp["weight_decay"],
            seed=cfg["seed"],
        )
        torch.save(model.state_dict(), ckpt_path)

    else:
        raise NotImplementedError(f"method {cfg['method']!r} not yet implemented")

    print(f"[task4] saved checkpoint -> {ckpt_path}")

    out_path = Path(cfg["output"]["results_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "method": cfg["method"],
        "seed": cfg["seed"],
        "history": history,
    }, indent=2))
    print(f"[task4] wrote {out_path}")


if __name__ == "__main__":
    main()
