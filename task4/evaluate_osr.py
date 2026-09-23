"""Task 4, Step 6: Common Evaluation.

Requires extract_outputs.py to have already been run for whichever tags
(vanilla, gcsc, proser) you want evaluated, caching features/logits/labels
under cache/{tag}_{split}.npz for CIFAR-10 train/val/test and the fixed
CIFAR-100 near/far unknown sets.

Produces:
  Table 1: MSP/MLS/Energy/Mahalanobis on the Vanilla model (near/far/all
           AUROC + validation-calibrated rejection).
  Table 2: Vanilla/GCSC/PROSER, CSA + near/far/all OSR metrics using MLS as
           the common score, PLUS an extra row for PROSER's own
           placeholder-based score.
  Failure cases: >=3 near-unknown and >=3 far-unknown examples accepted by
  the Vanilla+MLS calibrated threshold (i.e. false negatives for rejection),
  with predicted known class, true CIFAR-100 class name, score, threshold.

Usage (from inside task4/, after extract_outputs.py for each tag you want):
    python evaluate_osr.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "scores"))
sys.path.append(str(Path(__file__).resolve().parent / "evaluation"))
sys.path.append(str(Path(__file__).resolve().parent / "methods"))
sys.path.append(str(Path(__file__).resolve().parent / "data"))

from extract_outputs import load_cached  # noqa: E402
from msp import msp_score  # noqa: E402
from mls import mls_score  # noqa: E402
from energy import energy_score  # noqa: E402
from mahalanobis import fit_mahalanobis, mahalanobis_score  # noqa: E402
from metrics import osr_auroc_summary  # noqa: E402
from thresholds import calibrate_threshold, calibrated_rejection_summary  # noqa: E402
from proser import proser_placeholder_score  # noqa: E402
from cifar10 import CIFAR10_CLASSES  # noqa: E402


def top1_accuracy_from_logits(logits: np.ndarray, labels: np.ndarray, num_known: int = 10) -> float:
    preds = logits[:, :num_known].argmax(axis=1)
    return float((preds == labels).mean())


def compute_all_scores_for_tag(cache_dir: str, tag: str, num_known: int = 10) -> dict:
    """Computes MSP/MLS/Energy/Mahalanobis scores for a given cached tag,
    on val (for calibration) / test (known) / near / far.

    MSP/MLS/Energy are computed on ONLY the first num_known logit columns
    -- critical for PROSER, whose cached logits have num_known+num_dummy
    columns: without this slice, max_k/logsumexp would include the dummy
    classes, silently changing what these scores measure and breaking the
    "common score" comparison across Vanilla/GCSC/PROSER. Mahalanobis is
    unaffected (it uses features, not logits)."""
    train = load_cached(cache_dir, tag, "train")
    val = load_cached(cache_dir, tag, "val")
    test = load_cached(cache_dir, tag, "test")
    near = load_cached(cache_dir, tag, "near")
    far = load_cached(cache_dir, tag, "far")

    means, diag_var = fit_mahalanobis(
        train["features"].astype(np.float64), train["labels"], num_classes=num_known,
    )

    scores = {}
    for score_name, logit_fn, feat_fn in [
        ("msp", msp_score, None),
        ("mls", mls_score, None),
        ("energy", energy_score, None),
        ("mahalanobis", None, lambda f: mahalanobis_score(f.astype(np.float64), means, diag_var)),
    ]:
        if logit_fn is not None:
            scores[score_name] = {
                "val": logit_fn(val["logits"][:, :num_known]),
                "test": logit_fn(test["logits"][:, :num_known]),
                "near": logit_fn(near["logits"][:, :num_known]),
                "far": logit_fn(far["logits"][:, :num_known]),
            }
        else:
            scores[score_name] = {
                "val": feat_fn(val["features"]), "test": feat_fn(test["features"]),
                "near": feat_fn(near["features"]), "far": feat_fn(far["features"]),
            }

    return scores, {"train": train, "val": val, "test": test, "near": near, "far": far}


def build_score_table_row(scores: dict) -> dict:
    auroc = osr_auroc_summary(scores["test"], scores["near"], scores["far"])
    threshold = calibrate_threshold(scores["val"], percentile=95.0)
    rejection = calibrated_rejection_summary(threshold, scores["test"], scores["near"], scores["far"])
    return {**auroc, **rejection}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data_raw")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--num-known", type=int, default=10)
    ap.add_argument("--out", default="results/task4_final_comparison.json")
    args = ap.parse_args()

    results = {"table1_vanilla_scores": {}, "table2_model_comparison": {}, "failure_cases": {}}

    # --- Table 1: all 4 scores, Vanilla model only ---
    vanilla_path = Path(args.cache_dir) / "vanilla_test.npz"
    if vanilla_path.exists():
        print("\n[evaluate_osr] === Table 1: post-hoc scores on Vanilla ===")
        vanilla_scores, vanilla_cached = compute_all_scores_for_tag(args.cache_dir, "vanilla", args.num_known)
        for score_name, s in vanilla_scores.items():
            row = build_score_table_row(s)
            results["table1_vanilla_scores"][score_name] = row
            print(f"  {score_name}: AUROC(near/far/all)="
                  f"{row['auroc_known_vs_near']:.4f}/{row['auroc_known_vs_far']:.4f}/{row['auroc_known_vs_all']:.4f}  "
                  f"FPR@95TPR(near/far)={row['near_fpr_at_95tpr']:.4f}/{row['far_fpr_at_95tpr']:.4f}")
    else:
        print(f"[evaluate_osr] skipping Table 1: no cached vanilla outputs at {vanilla_path} "
              "(run extract_outputs.py --tag vanilla first)")
        vanilla_scores, vanilla_cached = None, None

    # --- Table 2: Vanilla / GCSC / PROSER comparison using MLS (+ PROSER's own score) ---
    print("\n[evaluate_osr] === Table 2: model comparison (MLS) ===")
    for tag in ["vanilla", "gcsc", "proser"]:
        tag_test_path = Path(args.cache_dir) / f"{tag}_test.npz"
        if not tag_test_path.exists():
            print(f"  skipping {tag!r}: no cached outputs at {tag_test_path}")
            continue

        if tag == "vanilla" and vanilla_scores is not None:
            scores, cached = vanilla_scores, vanilla_cached
        else:
            scores, cached = compute_all_scores_for_tag(args.cache_dir, tag, args.num_known)

        csa = top1_accuracy_from_logits(cached["test"]["logits"], cached["test"]["labels"], args.num_known)
        row = build_score_table_row(scores["mls"])
        row["closed_set_accuracy"] = csa
        results["table2_model_comparison"][tag] = row
        print(f"  {tag} (MLS): CSA={csa:.4f}  AUROC(near/far/all)="
              f"{row['auroc_known_vs_near']:.4f}/{row['auroc_known_vs_far']:.4f}/{row['auroc_known_vs_all']:.4f}")

        if tag == "proser":
            proser_own = {
                "val": proser_placeholder_score(cached["val"]["logits"], args.num_known),
                "test": proser_placeholder_score(cached["test"]["logits"], args.num_known),
                "near": proser_placeholder_score(cached["near"]["logits"], args.num_known),
                "far": proser_placeholder_score(cached["far"]["logits"], args.num_known),
            }
            proser_row = build_score_table_row(proser_own)
            proser_row["closed_set_accuracy"] = csa
            results["table2_model_comparison"]["proser_placeholder_score"] = proser_row
            print(f"  proser (placeholder score): CSA={csa:.4f}  AUROC(near/far/all)="
                  f"{proser_row['auroc_known_vs_near']:.4f}/{proser_row['auroc_known_vs_far']:.4f}/"
                  f"{proser_row['auroc_known_vs_all']:.4f}")

    # --- Failure cases: Vanilla + MLS, examples accepted despite being unknown ---
    if vanilla_scores is not None:
        print("\n[evaluate_osr] === Failure case inspection (Vanilla + MLS) ===")
        cifar100_classes = torchvision.datasets.CIFAR100(
            root=args.data_root, train=False, download=True,
        ).classes

        threshold = calibrate_threshold(vanilla_scores["mls"]["val"], percentile=95.0)
        for split in ["near", "far"]:
            s = vanilla_scores["mls"][split]
            cached_split = vanilla_cached[split]
            accepted_mask = s <= threshold  # false negatives: unknown but accepted
            accepted_idx = np.where(accepted_mask)[0]
            n_show = min(5, len(accepted_idx))
            cases = []
            for i in accepted_idx[:n_show]:
                pred_class_idx = int(cached_split["logits"][i, :args.num_known].argmax())
                true_fine_label = int(cached_split["labels"][i])
                cases.append({
                    "true_cifar100_class": cifar100_classes[true_fine_label],
                    "predicted_known_class": CIFAR10_CLASSES[pred_class_idx],
                    "score": float(s[i]),
                    "threshold": threshold,
                })
            results["failure_cases"][split] = cases
            print(f"  {split}: {len(cases)} example(s) shown (of {int(accepted_mask.sum())} total false negatives)")
            for c in cases:
                print(f"    true={c['true_cifar100_class']!r} -> predicted={c['predicted_known_class']!r}  "
                      f"score={c['score']:.4f} (threshold={c['threshold']:.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[evaluate_osr] wrote {out_path}")


if __name__ == "__main__":
    main()
