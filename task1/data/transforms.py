"""Task 1 intervention transforms.

Per the assignment: "Construct interventions on a common 224x224 RGB image
before applying each model's required normalization." So every function here
operates on a PIL Image (or a [0,1] float tensor, C,H,W) at 224x224, RGB,
*before* any backbone-specific `preprocess` (resize/crop/normalize) is
applied. Each backbone's own `preprocess` is applied afterward, separately,
in the evaluation script.

Implemented here:
  - to_base_224: shared resize to the common 224x224 canvas.
  - grayscale: removes color, preserves geometry (required common color intervention).
  - hue_rotate: fixed hue rotation (one of the three allowed additional color
    interventions) — changes chromatic identity, preserves luminance/shape.
  - translate_reflect: reflection-padded shift by (dx, dy) pixels, used for
    the 0/8/16/32-pixel, four-cardinal-direction translation experiment.
  - patch_shuffle: one non-identity 4x4 grid permutation per image, seeded.

Note: the shape/texture cue-conflict images (AdaIN) are intentionally NOT
here — that's a separate, heavier pipeline (`make_cue_conflicts.py`, to be
added) since it needs a style-transfer model rather than a simple pixel op.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

BASE_SIZE = 224


def to_base_224(img: Image.Image) -> Image.Image:
    """Resize (not crop) to the shared 224x224 canvas that every subsequent
    intervention and every backbone receives identically."""
    return img.resize((BASE_SIZE, BASE_SIZE), Image.BICUBIC)


def grayscale(img: Image.Image) -> Image.Image:
    """Removes color, preserves object geometry. Output stays 3-channel
    (duplicated) so it's a drop-in replacement for any backbone's expected
    RGB input."""
    return TF.to_pil_image(TF.rgb_to_grayscale(TF.to_tensor(img), num_output_channels=3))


def hue_rotate(img: Image.Image, degrees: float = 90.0) -> Image.Image:
    """Fixed hue rotation: shifts chromatic identity by a constant amount
    while preserving saturation/lightness structure and object geometry.
    `degrees` is converted to torchvision's [-0.5, 0.5] hue-factor range
    (which represents a full 360-degree rotation)."""
    hue_factor = (degrees % 360) / 360.0
    if hue_factor > 0.5:
        hue_factor -= 1.0
    return TF.adjust_hue(img, hue_factor)


def translate_reflect(img: Image.Image, dx: int, dy: int) -> Image.Image:
    """Shift the image by (dx, dy) pixels using reflection padding followed
    by a shifted crop back to the original size, per the assignment's
    specified translation procedure.

    Positive dx -> content shifts right; positive dy -> content shifts down.
    (dx, dy) = (0, 0) is the identity / clean case.
    """
    if dx == 0 and dy == 0:
        return img
    tensor = TF.to_tensor(img).unsqueeze(0)  # (1, C, H, W)
    pad = max(abs(dx), abs(dy))
    padded = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    # Crop a BASE_SIZE x BASE_SIZE window offset by (dx, dy) from center.
    top = pad - dy
    left = pad - dx
    cropped = padded[:, :, top:top + BASE_SIZE, left:left + BASE_SIZE]
    return TF.to_pil_image(cropped.squeeze(0))


CARDINAL_DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


def translations_for_displacement(img: Image.Image, delta: int) -> dict[str, Image.Image]:
    """All four cardinal-direction translations at a given pixel displacement
    `delta` (0, 8, 16, or 32 per the assignment). Average results across
    these four when computing accuracy/consistency at each delta."""
    if delta == 0:
        return {"up": img, "down": img, "left": img, "right": img}
    out = {}
    for name, (sx, sy) in CARDINAL_DIRECTIONS.items():
        out[name] = translate_reflect(img, dx=sx * delta, dy=sy * delta)
    return out


def patch_shuffle(img: Image.Image, seed: int, grid: int = 4) -> tuple[Image.Image, int]:
    """Divide the image into a `grid` x `grid` pixel-space grid and apply one
    non-identity random permutation of the patches, deterministic given
    `seed` (which should be derived per-image so different images get
    different, but reproducible, permutations while still using the master
    seed 6304 as the base).

    Returns (shuffled_image, permutation_id) where permutation_id can be
    logged for reproducibility/debugging.
    """
    assert BASE_SIZE % grid == 0
    patch = BASE_SIZE // grid
    arr = np.array(img)  # (H, W, C)

    rng = np.random.default_rng(seed)
    n_patches = grid * grid
    perm = rng.permutation(n_patches)
    # Reject the identity permutation (guaranteed non-identity per assignment).
    attempts = 0
    while np.array_equal(perm, np.arange(n_patches)) and attempts < 10:
        perm = rng.permutation(n_patches)
        attempts += 1

    patches = [
        arr[(i // grid) * patch:(i // grid + 1) * patch,
            (i % grid) * patch:(i % grid + 1) * patch]
        for i in range(n_patches)
    ]
    shuffled = np.zeros_like(arr)
    for dst_idx, src_idx in enumerate(perm):
        r, c = dst_idx // grid, dst_idx % grid
        shuffled[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = patches[src_idx]

    perm_id = int("".join(str(p) for p in perm)) if n_patches <= 16 else hash(tuple(perm.tolist()))
    return Image.fromarray(shuffled), perm_id
