# Task 1 — Inductive Biases and Feature Representations

## Status
- [x] `data/make_subset.py` — stratified 80/20 train/val split + class-balanced
      500-image eval subset, seed 6304, saved as index JSON (no images duplicated).
- [x] `data/transforms.py` — shared 224x224 canvas, grayscale, hue rotation,
      reflection-pad translation (0/8/16/32 px, 4 directions), 4x4 patch shuffle.
- [x] `models/backbones.py` — frozen ResNet-50 (IMAGENET1K_V2), ViT-B/16
      (IMAGENET1K_V1), CLIP ViT-B-32 (openai) wrappers + `LinearHead`.
- [x] `scripts/checkpoints.py` — save/load trained heads + predictions so
      later steps reuse Step 1's heads instead of retraining.
- [x] `scripts/run_task1_clean_baseline.py` — now also saves head checkpoints
      (`results/checkpoints/<backbone>_head.pt`) and clean predictions
      (`results/predictions/clean_<backbone>.npz`, `clean_clip_zero_shot.npz`).
- [x] `data/transformed_dataset.py` — generic wrapper applying any
      PIL->PIL transform to a base dataset.
- [x] `scripts/run_task1_color_bias.py` — Step 2: grayscale + fixed hue
      rotation, reusing Step 1's heads; reports accuracy delta + prediction
      consistency vs. clean.
- [ ] `data/make_cue_conflicts.py` — AdaIN-based shape/texture cue conflicts
      (>=5 class pairs, >=200 valid conflicts, rejection rule).
- [ ] `scripts/run_task1_translation.py` — Step 4.
- [ ] `scripts/run_task1_patch_shuffle.py` — Step 5.
- [ ] `analysis/evaluate_bias.py`, `analysis/feature_similarity.py`,
      `analysis/representation.py` — Step 6.

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
python data/make_subset.py --dataset stl10 --data-root ./data_raw --out-dir results/splits
python scripts/run_task1_clean_baseline.py --config configs/clean_baseline.yaml
python scripts/run_task1_color_bias.py --config configs/clean_baseline.yaml
```

Step 1 writes `results/splits/...json`, `results/checkpoints/<backbone>_head.pt`,
`results/predictions/clean_*.npz`, and `results/clean_baseline_seed6304.json`.
Step 2 reuses the saved heads/predictions and writes `results/color_bias_seed6304.json`
plus `results/predictions/{grayscale,hue_rotate_90}_*.npz` (kept for Step 6's
representation analysis, which needs the same transformed images' features).

**Not yet run end-to-end** — no GPU/disk available in the sandbox this was
written in. Run it yourself and report back anything that breaks (most likely
spot: the CLIP zero-shot DataLoader's collate_fn, or feature_dim mismatches).

## Next steps
3. AdaIN cue-conflict generation (external code, to be attributed in this README).
4. Translation sweep (0/8/16/32px, 4 directions) using `transforms.py`'s
   `translations_for_displacement`, same reuse-heads pattern as color bias.
5. Patch shuffle (Step 5), same pattern.
6. Representation analysis: cosine stability I_T (need to also save
   *features*, not just predictions, for grayscale/cue-conflict/translation/
   patch-shuffle — current color_bias script only persists predictions) and
   t-SNE/UMAP plots.