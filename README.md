# Beyond IID and Closed-Set Learning: Inductive Biases, Domain Shift, and Open-Set Recognition

EE-5102/CS-6304 Advanced Topics in Machine Learning, Fall 2026, Programming Assignment 1

**Author:** Zarman Ali Awan (28100092)
**Report:** [`report/main.pdf`](report/main.pdf)

This repository contains code, configurations, and results for four experiments studying model behavior beyond the standard IID, closed-set assumption:

- **Task 1, Inductive Biases and Feature Representations:** comparing ResNet-50, ViT-B/16, and CLIP under controlled visual interventions (color, cue-conflict, translation, patch shuffle) on STL-10.
- **Task 2, Unsupervised Domain Adaptation:** Source-only, DAN, DANN, and CDAN on PACS (Photo/Art/Cartoon to Sketch).
- **Task 3, Domain Generalization:** ERM, DAN-DG, and SAM on PACS, with the Sketch target domain fully withheld from training.
- **Task 4, Open-Set Recognition:** Vanilla, GCSC, and PROSER on CIFAR-10 (known) vs. a fixed CIFAR-100 near/far unknown split, using MSP/MLS/Energy/Mahalanobis post-hoc scores.

## Repository Structure

```
pa1-beyond-iid/
├── README.md
├── requirements.txt
├── .gitignore
├── common/                  # shared utilities (seeding, metrics, plotting)
├── shared/                  # shared PACS loading/splitting logic (Tasks 2-3)
│   ├── pacs.py
│   ├── pacs_protocol.py
│   └── splits/pacs_sketch_seed6304.json
├── task1/
│   ├── configs/
│   ├── data/                # subset selection, cue-conflict generation, transforms
│   ├── models/               # ResNet/ViT/CLIP backbone wrappers
│   ├── analysis/              # shape-bias, feature-similarity, representation analysis
│   ├── scripts/run_task1.py
│   └── results/
├── task2/
│   ├── configs/               # source_only.yaml, dan.yaml, dann.yaml, cdan.yaml, dan_lambda*.yaml
│   ├── models/
│   ├── methods/
│   ├── evaluation/
│   ├── train.py
│   ├── evaluate_target.py
│   ├── evaluate_final.py
│   ├── evaluate_dan_controlled_study.py
│   └── results/
├── task3/
│   ├── configs/               # erm.yaml, dan_dg.yaml, sam.yaml, dan_dg_lambda*.yaml
│   ├── models/ methods/ evaluation/
│   ├── train.py
│   ├── evaluate_sketch.py
│   ├── evaluate_controlled_study.py
│   └── results/
├── task4/
│   ├── configs/               # vanilla.yaml, gcsc.yaml, proser.yaml
│   ├── data/ models/ methods/ scores/ evaluation/
│   ├── train.py
│   ├── extract_outputs.py
│   ├── evaluate_osr.py
│   └── results/
└── report/
    ├── main.tex
    ├── main.pdf
    ├── references.bib
    ├── neurips_2026.sty
    └── figures/
```

Each task's `results/` directory contains machine-readable JSON files (per-epoch training history, final metrics, controlled-study sweeps) that every number in the report traces back to directly.

## Environment Setup

```bash
git clone https://github.com/alizarman786-prog/atml-assignment-1.git
cd atml-assignment-1
pip install -r requirements.txt
pip install scipy scikit-learn matplotlib
```

All experiments were run on a single GPU (T4/P100 class). Task 1 uses STL-10 and OpenCLIP; Tasks 2 and 3 use PACS (auto-downloaded via HuggingFace `flwrlabs/pacs`); Task 4 uses CIFAR-10/CIFAR-100 (auto-downloaded via `torchvision`).

A fixed seed of **6304** is used throughout for splitting, subset selection, and comparisons.

## Reproducing Results

### Task 1
```bash
cd task1
python scripts/run_task1.py --device cuda
```
Runs the clean baseline, color-bias, shape/texture cue-conflict, translation, patch-shuffle, and representation-analysis steps in sequence, saving all metrics and figures under `results/`.

### Task 2
```bash
cd task2
python train.py --config configs/source_only.yaml --device cuda
python evaluate_target.py --config configs/source_only.yaml --device cuda
python train.py --config configs/dan.yaml --device cuda
python evaluate_target.py --config configs/dan.yaml --device cuda
python train.py --config configs/dann.yaml --device cuda
python evaluate_target.py --config configs/dann.yaml --device cuda
python train.py --config configs/cdan.yaml --device cuda
python evaluate_target.py --config configs/cdan.yaml --device cuda
python evaluate_final.py --config configs/source_only.yaml --device cuda
python train.py --config configs/dan_lambda01.yaml --device cuda
python train.py --config configs/dan_lambda10.yaml --device cuda
python evaluate_dan_controlled_study.py --device cuda
```

### Task 3
```bash
cd task3
python train.py --config configs/erm.yaml --device cuda      # reuses Task 2's Source-only checkpoint
python train.py --config configs/dan_dg.yaml --device cuda
python train.py --config configs/sam.yaml --device cuda
python evaluate_sketch.py --device cuda
python train.py --config configs/dan_dg_lambda01.yaml --device cuda
python train.py --config configs/dan_dg_lambda10.yaml --device cuda
python evaluate_controlled_study.py --device cuda
```

### Task 4
```bash
cd task4
python train.py --config configs/vanilla.yaml --device cuda
python extract_outputs.py --checkpoint results/checkpoints/vanilla.pt --tag vanilla --device cuda
python train.py --config configs/gcsc.yaml --device cuda
python extract_outputs.py --checkpoint results/checkpoints/gcsc.pt --tag gcsc --device cuda
python train.py --config configs/proser.yaml --device cuda
python extract_outputs.py --checkpoint results/checkpoints/proser.pt --tag proser --device cuda
python evaluate_osr.py
```

## Report

The full report (`report/main.tex` / `report/main.pdf`) covers Setup, Results, and Discussion for all four tasks plus a Cross-Task Synthesis. Every table and figure is generated directly from the JSON result files in each task's `results/` directory, and every figure's generating script is included alongside it.

## Attribution

- Backbones: `torchvision` (ResNet-50, ViT-B/16, ResNet-18) and OpenCLIP (`ViT-B-32`, `pretrained='openai'`), used via their public pretrained weights.
- Cue-conflict stylization uses the authors' pretrained AdaIN weights (Huang and Belongie, 2017).
- PROSER's classifier/data-placeholder losses follow Zhou et al. (2021); our implementation is our own, guided by the paper's formulation.
- Coding assistance from an LLM (Claude) was used during development, per the assignment's permitted-use policy for code; all submitted code was reviewed and is understood by the author. Report writing, analysis, and interpretation are the author's own.

## Notes

- Checkpoints (`*.pt`) and downloaded datasets are intentionally excluded from this repository via `.gitignore`; re-running the commands above regenerates them.
- `task4/cache/*.npz` (extracted features/logits) are also gitignored and regenerated by `extract_outputs.py`.
