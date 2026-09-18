"""t-SNE visualization of clean vs. transformed representations, per backbone,
for Task 1 Step 6."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402


def plot_tsne_clean_vs_transformed(
    clean_feats: np.ndarray, transformed_feats: np.ndarray, labels: np.ndarray,
    class_names: list[str], title: str, out_path: str, seed: int = 6304, perplexity: int = 30,
):
    """Fits ONE 2D t-SNE projection to the combined clean+transformed features
    (same subset, same seed, per the assignment), colors by ground-truth
    class, and distinguishes clean ('o') from transformed ('x') by marker."""
    combined = np.concatenate([clean_feats, transformed_feats], axis=0)
    condition = np.array(["clean"] * len(clean_feats) + ["transformed"] * len(transformed_feats))
    combined_labels = np.concatenate([labels, labels])

    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    proj = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")
    for cls_idx in range(len(class_names)):
        for cond, marker in [("clean", "o"), ("transformed", "x")]:
            mask = (combined_labels == cls_idx) & (condition == cond)
            ax.scatter(proj[mask, 0], proj[mask, 1], color=cmap(cls_idx % 10),
                       marker=marker, s=18, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    legend_elems = [
        Line2D([0], [0], marker="o", color="gray", linestyle="", label="clean"),
        Line2D([0], [0], marker="x", color="gray", linestyle="", label="transformed"),
    ]
    ax.legend(handles=legend_elems, loc="best")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return proj
