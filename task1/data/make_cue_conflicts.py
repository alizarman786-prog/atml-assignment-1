"""Task 1, Step 3: Shape vs. Texture cue-conflict generation.

For >=5 unordered class pairs, generates cue-conflict images in both
directions (shape/content=A,texture/style=B and vice versa) using AdaIN
style transfer (alpha=1.0, full style transfer) on images drawn from the
Task 1 eval subset. Applies a visual rejection rule DEFINED BEFORE looking at
any model predictions (per the assignment's explicit requirement not to use
model predictions to decide which images to retain): reject an output if it
is degenerate (near-constant color / collapsed) or contains non-finite
values, both symptomatic of a failed stylization rather than a valid but
unusual-looking image.

Saves accepted stylized images as PNGs plus a JSON manifest recording, per
image: content class, style class, direction, and the rejection counts per
class-pair/direction combo (so the accepted/rejected counts required by the
assignment are auditable).

Usage (run from inside task1/, same as the other scripts):
    python data/make_cue_conflicts.py --config configs/clean_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import yaml

sys.path.append(str(Path(__file__).resolve().parent))         # task1/data
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))  # task1/models
sys.path.append(str(Path(__file__).resolve().parents[2]))     # repo root

from dataset import build_split_datasets  # noqa: E402
from adain import AdaINStyleTransfer  # noqa: E402
from common.seed import seeded_rng  # noqa: E402

SEED = 6304
TARGET_PER_COMBO = 25          # 5 pairs x 2 directions x 25 = 625 attempts;
MIN_TOTAL_VALID = 200          # assignment's minimum
MAX_ATTEMPTS_PER_COMBO = 60    # generous ceiling if rejection rate is high
DEGENERATE_STD_THRESHOLD = 0.03  # reject near-constant-color outputs

# Five unordered class pairs with clear, interpretable shape/texture
# conflicts. Chosen for visually distinct textures per pair (fur vs. fur
# pattern, metal vs. metal, organic vs. mechanical) so a human can judge
# "shape of A, texture of B" by eye. Class NAMES (resolved to indices at
# runtime against the actual dataset's class list).
CLASS_PAIRS = [
    ("cat", "dog"),
    ("deer", "horse"),
    ("airplane", "ship"),
    ("car", "truck"),
    ("bird", "monkey"),
]


def is_valid_stylization(img_tensor: torch.Tensor) -> bool:
    """The visual rejection rule, fixed before any model evaluation. Rejects
    outputs that are non-finite or have collapsed to near-constant color
    (both are symptoms of a failed AdaIN decode, not a legitimately unusual
    but valid cue-conflict image)."""
    if not torch.isfinite(img_tensor).all():
        return False
    if img_tensor.std().item() < DEGENERATE_STD_THRESHOLD:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="data_raw/cue_conflicts")
    ap.add_argument("--manifest-out", default="results/cue_conflicts_manifest.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    _, _, eval_ds = build_split_datasets(cfg["dataset"], cfg["data_root"], cfg["splits_path"])
    class_names = eval_ds.class_names
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    # Index eval_ds examples by class for fast sampling.
    indices_by_class: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
    for i in range(len(eval_ds)):
        _, label = eval_ds[i]
        indices_by_class[label].append(i)

    print("[cue_conflicts] loading AdaIN model (downloads pretrained weights on first use)...")
    adain = AdaINStyleTransfer(cache_dir=str(Path(cfg["data_root"]) / "adain_weights"))
    adain.to(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = seeded_rng(SEED)
    manifest = []
    combo_stats = {}
    total_accepted = 0

    directions = []
    for a, b in CLASS_PAIRS:
        directions.append((a, b))  # shape/content=a, texture/style=b
        directions.append((b, a))  # shape/content=b, texture/style=a

    for content_name, style_name in directions:
        combo_key = f"{content_name}_shape__{style_name}_texture"
        content_idx = name_to_idx[content_name]
        style_idx = name_to_idx[style_name]
        content_pool = indices_by_class[content_idx]
        style_pool = indices_by_class[style_idx]

        accepted, rejected, attempts = 0, 0, 0
        while accepted < TARGET_PER_COMBO and attempts < MAX_ATTEMPTS_PER_COMBO:
            attempts += 1
            c_i = content_pool[rng.integers(0, len(content_pool))]
            s_i = style_pool[rng.integers(0, len(style_pool))]
            content_img, _ = eval_ds[c_i]
            style_img, _ = eval_ds[s_i]

            content_t = TF.to_tensor(content_img).unsqueeze(0).to(args.device)
            style_t = TF.to_tensor(style_img).unsqueeze(0).to(args.device)

            out_t = adain.style_transfer(content_t, style_t, alpha=1.0).squeeze(0).cpu()

            if not is_valid_stylization(out_t):
                rejected += 1
                continue

            fname = f"{combo_key}_{accepted:03d}.png"
            out_path = out_dir / fname
            TF.to_pil_image(out_t).save(out_path)

            manifest.append({
                "filename": fname,
                "content_class": content_name,
                "content_label": content_idx,
                "style_class": style_name,
                "style_label": style_idx,
                "content_eval_idx": c_i,
                "style_eval_idx": s_i,
            })
            accepted += 1

        combo_stats[combo_key] = {"accepted": accepted, "rejected": rejected, "attempts": attempts}
        total_accepted += accepted
        print(f"  [{combo_key}] accepted={accepted} rejected={rejected} attempts={attempts}")

    print(f"\n[cue_conflicts] total accepted={total_accepted} "
          f"(minimum required: {MIN_TOTAL_VALID})")
    if total_accepted < MIN_TOTAL_VALID:
        print("  WARNING: below the assignment's minimum of 200 valid conflicts. "
              "Consider raising MAX_ATTEMPTS_PER_COMBO or lowering DEGENERATE_STD_THRESHOLD "
              "(document any threshold change in the report).")

    manifest_record = {
        "seed": SEED,
        "dataset": cfg["dataset"],
        "class_pairs": CLASS_PAIRS,
        "rejection_rule": f"reject if non-finite or std < {DEGENERATE_STD_THRESHOLD}",
        "combo_stats": combo_stats,
        "total_accepted": total_accepted,
        "images": manifest,
    }
    out_manifest_path = Path(args.manifest_out)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_path.write_text(json.dumps(manifest_record, indent=2))
    print(f"[cue_conflicts] wrote {out_manifest_path}")
    print(f"[cue_conflicts] wrote {total_accepted} images to {out_dir}")


if __name__ == "__main__":
    main()
