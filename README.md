# PA1 — Beyond IID, Closed-Set Learning

EE-5102/CS-6304 Advanced Topics in Machine Learning, Fall 2026 — Programming Assignment 1.

Four tasks studying learning beyond the IID, closed-set assumption:

- **Task 1** — Inductive Biases & Feature Representations (ResNet-50 / ViT-B/16 / CLIP, cue-conflict, translation, patch shuffle)
- **Task 2** — Unsupervised Domain Adaptation (PACS, DAN / DANN / CDAN)
- **Task 3** — Domain Generalization (PACS, ERM / DAN-DG / SAM)
- **Task 4** — Open-Set Recognition (CIFAR-10 known / CIFAR-100 unknown, MSP / MLS / Energy / Mahalanobis, GCSC, PROSER)

Global seed for all reported comparisons: **6304**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```
common/        shared utilities: seeding, logging, plotting, metrics
shared/        PACS loading + protocol shared by task2/task3
task1/         inductive biases & representations
task2/         unsupervised domain adaptation
task3/         domain generalization
task4/         open-set recognition
report/        NeurIPS-format PDF report source + figures
```

Each task directory follows: `configs/` (yaml experiment configs), `data/` or dataset scripts,
`models/` (architecture wrappers), `methods/` (training objectives), `evaluation/` (metrics,
diagnostics), `train.py` / `run_task*.py` (entry points), `results/` (saved JSON/CSV outputs —
small files only, no checkpoints).

## Reproducing Task 1

```bash
cd task1
python data/make_subset.py --dataset stl10 --data-root ./data_raw --out-dir results/splits
python scripts/run_task1.py --config configs/clean_baseline.yaml
```

See each task's own README (to be added as that task is completed) for full reproduction commands.

## Data

Raw datasets are **not** committed. Note the naming: each task's `data/` folder holds *source
code* (e.g. `task1/data/make_subset.py`), while downloaded dataset files are cached separately
under a gitignored `data_raw/` folder (default `--data-root ./data_raw`) so the two never
collide. PACS lives under `shared/data_raw/`, CIFAR-10/100 under `task4/data_raw/`.

## Attribution

External code (e.g. AdaIN style transfer, PROSER/RPL reference implementations) used or adapted
is documented in the relevant task's README with a link to the source.
