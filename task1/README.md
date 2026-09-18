# Task 1 — Inductive Biases and Feature Representations

## Status
- [x] `data/make_subset.py` — stratified 80/20 train/val split + class-balanced
      500-image eval subset, seed 6304, saved as index JSON (no images duplicated).
- [x] `data/transforms.py` — shared 224x224 canvas, grayscale, hue rotation,
      reflection-pad translation (0/8/16/32 px, 4 directions), 4x4 patch shuffle.
- [x] `models/backbones.py` — frozen ResNet-50 (IMAGENET1K_V2), ViT-B/16
      (IMAGENET1K_V1), CLIP ViT-B-32 (openai) wrappers + `LinearHead`.
- [ ] `data/make_cue_conflicts.py` — AdaIN-based shape/texture cue conflicts
      (>=5 class pairs, >=200 valid conflicts, rejection rule).
- [ ] `scripts/run_task1.py` — orchestrates: train heads -> clean baseline ->
      color bias -> shape/texture -> translation -> patch shuffle -> representation analysis.
- [ ] `analysis/evaluate_bias.py` — shape bias / coverage computation.
- [ ] `analysis/feature_similarity.py` — cosine stability I_T.
- [ ] `analysis/representation.py` — t-SNE/UMAP visualization.

## Design choices (fill in and justify in the report)
- **Dataset:** STL-10 (recommended) vs Oxford-IIIT Pets — pick one, state why.
- **Cue-conflict class pairs & style strength:** to be fixed before running AdaIN.
- **Additional color intervention:** grayscale is required; choose one of
  {fixed hue rotation, palette transfer, class-swapped color stats}. Currently
  `hue_rotate` is implemented as the fixed-hue-rotation option.
- **Representation visualization:** t-SNE or UMAP, and its hyperparameters
  (perplexity/neighbors, seed=6304, same subset across backbones).

State a hypothesis + metric for each of these before looking at results
(per assignment instructions).

## Reproduce so far

```bash
cd task1
python data/make_subset.py --dataset stl10 --data-root ./data --out-dir results/splits
```

This downloads STL-10 (if not already present under `./data`) and writes
`results/splits/stl10_splits_seed6304.json` with `train_idx`, `val_idx`, and
`eval_subset_idx`.

## Next steps
1. Implement `models/backbones.py`'s `LinearHead` training loop (AdamW,
   lr=1e-3, wd=1e-4, ≤50 epochs, early stop patience 5) → clean baseline table
   (top-1, macro-F1, mean max confidence) for all 3 heads + zero-shot CLIP.
2. Color bias: apply `grayscale` and `hue_rotate` to the eval subset, rerun
   all 4 models, report accuracy delta + prediction consistency vs. clean.
3. AdaIN cue-conflict generation (external code, to be attributed in this README).
4. Translation and patch-shuffle sweeps using `transforms.py`.
5. Representation analysis tying predicted-class changes to cosine stability
   and t-SNE/UMAP plots.
