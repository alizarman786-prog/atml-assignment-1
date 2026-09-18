"""Task 1, Step 6: Representation Analysis.

For grayscale, cue-conflict, translation (32px, averaged over 4 directions),
and patch-shuffle, computes the cosine representation-stability index I_T
between each backbone's clean and transformed features. Also fits one t-SNE
projection per backbone to the combined clean + patch-shuffled features
(patch-shuffle chosen as the representative "transformed" condition for
visualization since it is the most disruptive intervention) and saves it as
a figure.

Requires run_task1_clean_baseline.py and make_cue_conflicts.py to have been
run first.

Usage (from inside task1/):
    python scripts/run_task1_representation.py --config configs/clean_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dataset import build_split_datasets  # noqa: E402
from transformed_dataset import TransformedDataset  # noqa: E402
from in_memory_dataset import InMemoryDataset  # noqa: E402
from cue_conflict_dataset import CueConflictDataset  # noqa: E402
from transforms import grayscale, CARDINAL_DIRECTIONS, translate_reflect, patch_shuffle  # noqa: E402
from backbones import build_backbone  # noqa: E402
from extract_features import extract_features  # noqa: E402
from feature_similarity import representation_stability  # noqa: E402
from representation import plot_tsne_clean_vs_transformed  # noqa: E402
from common.seed import set_seed  # noqa: E402
from common.metrics import cosine_similarity_pairs  # noqa: E402

SEED = 6304
GRID = 4
TRANSLATION_DELTA = 32


class _LabelAdapter(Dataset):
    """CueConflictDataset yields (img, content_label, style_label, filename);
    extract_features expects (img, label) pairs. Wrap it so the content
    label rides along as "label" (unused for I_T, only order matters)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, content_label, _style_label, _fname = self.base[i]
        return img, content_label


def build_patch_shuffled_dataset(eval_ds) -> InMemoryDataset:
    items = []
    for i in range(len(eval_ds)):
        img, label = eval_ds[i]
        shuffled_img, _ = patch_shuffle(img, seed=SEED + i, grid=GRID)
        items.append((shuffled_img, label))
    return InMemoryDataset(items, class_names=eval_ds.class_names)


def make_translate_fn(dx_sign: int, dy_sign: int, delta: int):
    def _fn(img):
        return translate_reflect(img, dx=dx_sign * delta, dy=dy_sign * delta)
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default="results/cue_conflicts_manifest.json")
    ap.add_argument("--images-dir", default="data_raw/cue_conflicts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--figures-dir", default="results/figures")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])
    results_dir = Path(cfg["output"]["results_json"]).parent

    _, _, eval_ds = build_split_datasets(cfg["dataset"], cfg["data_root"], cfg["splits_path"])
    class_names = eval_ds.class_names
    eval_labels_all = np.array([eval_ds[i][1] for i in range(len(eval_ds))])

    conflict_ds = CueConflictDataset(args.manifest, args.images_dir)
    conflict_content_idx = np.array([item["content_eval_idx"] for item in conflict_ds.items])
    conflict_adapter = _LabelAdapter(conflict_ds)

    patch_ds = build_patch_shuffled_dataset(eval_ds)

    results = {"dataset": cfg["dataset"], "seed": cfg["seed"],
               "translation_delta_used": TRANSLATION_DELTA, "models": {}}

    for name in cfg["backbones"]:
        print(f"\n[representation] === {name} ===")
        backbone = build_backbone(name, class_names=class_names if name == "clip_vit_b_32" else None)
        backbone.to(args.device)

        clean_feats, _ = extract_features(backbone, eval_ds, device=args.device)
        clean_feats = clean_feats.numpy()

        # Grayscale
        gray_ds = TransformedDataset(eval_ds, grayscale)
        gray_feats, _ = extract_features(backbone, gray_ds, device=args.device)
        i_t_gray = representation_stability(clean_feats, gray_feats.numpy())
        print(f"  I_T grayscale:    {i_t_gray:.4f}")

        # Translation (32px, averaged over 4 directions)
        trans_i_ts = []
        for direction, (sx, sy) in CARDINAL_DIRECTIONS.items():
            trans_ds = TransformedDataset(eval_ds, make_translate_fn(sx, sy, TRANSLATION_DELTA))
            trans_feats, _ = extract_features(backbone, trans_ds, device=args.device)
            trans_i_ts.append(representation_stability(clean_feats, trans_feats.numpy()))
        i_t_translation = float(np.mean(trans_i_ts))
        print(f"  I_T translation ({TRANSLATION_DELTA}px, avg over 4 dir): {i_t_translation:.4f}")

        # Patch shuffle
        patch_feats, _ = extract_features(backbone, patch_ds, device=args.device)
        patch_feats_np = patch_feats.numpy()
        i_t_patch = representation_stability(clean_feats, patch_feats_np)
        print(f"  I_T patch_shuffle: {i_t_patch:.4f}")

        # Cue conflict: pair each conflict image's feature with its OWN
        # content-source clean image's feature (not eval_ds[i] elementwise).
        conflict_feats, _ = extract_features(backbone, conflict_adapter, device=args.device)
        paired_clean = clean_feats[conflict_content_idx]
        i_t_cue_conflict = float(np.mean(cosine_similarity_pairs(paired_clean, conflict_feats.numpy())))
        print(f"  I_T cue_conflict: {i_t_cue_conflict:.4f}")

        results["models"][name] = {
            "i_t_grayscale": i_t_gray,
            "i_t_translation_32px_avg": i_t_translation,
            "i_t_patch_shuffle": i_t_patch,
            "i_t_cue_conflict": i_t_cue_conflict,
        }

        # t-SNE: clean vs. patch-shuffled (most disruptive -> most informative plot)
        fig_path = Path(args.figures_dir) / f"tsne_{name}_clean_vs_patchshuffle.png"
        plot_tsne_clean_vs_transformed(
            clean_feats, patch_feats_np, eval_labels_all, class_names,
            title=f"{name}: clean vs. patch-shuffled (t-SNE)",
            out_path=str(fig_path), seed=cfg["seed"], perplexity=30,
        )
        print(f"  wrote {fig_path}")
        results["models"][name]["tsne_figure"] = str(fig_path)

    out_path = results_dir / "representation_analysis_seed6304.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[representation] wrote {out_path}")


if __name__ == "__main__":
    main()
