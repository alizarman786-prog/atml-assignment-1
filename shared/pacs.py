"""PACS dataset loading, shared by Task 2 and Task 3.

Uses the flwrlabs/pacs HuggingFace dataset (a faithful mirror of the
official PACS release -- Photo 1670, Art Painting 2048, Cartoon 2344,
Sketch 3929 images, summing to the official 9991 total), loaded via
`datasets.load_dataset`, which avoids the Google-Drive-quota fragility of
older PACS download scripts. No authentication required.

Domains: photo, art_painting, cartoon, sketch.
Classes (7): dog, elephant, giraffe, guitar, horse, house, person
(exact integer<->name mapping taken from the HF dataset's own ClassLabel
feature, not hardcoded, so it is always correct regardless of ordering).
"""
from __future__ import annotations

import torchvision.transforms as T
from torch.utils.data import Dataset

PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]

# ImageNet normalization, matching ResNet18_Weights.IMAGENET1K_V1's expected
# input statistics (the assignment specifies "the normalization associated
# with the pretrained weights").
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Per the assignment: "Resize images to 256x256 and use a random 224x224
# crop with horizontal flipping during training; use a 224x224 center crop
# for validation and evaluation."
PACS_TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

PACS_EVAL_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def load_pacs_hf(cache_dir: str | None = None):
    """Downloads (first call only, cached by the `datasets` library) and
    returns the full PACS HuggingFace Dataset object (single 'train' split
    containing all 9991 images across all 4 domains, with 'image', 'domain',
    'label' columns)."""
    from datasets import load_dataset

    ds = load_dataset("flwrlabs/pacs", cache_dir=cache_dir)
    return ds["train"]


def get_domain_indices(hf_dataset, domain_name: str) -> list[int]:
    """Global indices (into hf_dataset) of every example belonging to the
    given domain. Computed by a single linear scan of the 'domain' column
    (fast -- it's just strings, no image decoding)."""
    domains = hf_dataset["domain"]
    return [i for i, d in enumerate(domains) if d == domain_name]


def get_domain_labels(hf_dataset, indices: list[int]) -> list[int]:
    """Integer class labels for the given global indices, without decoding
    any images."""
    all_labels = hf_dataset["label"]
    return [all_labels[i] for i in indices]


class PACSSubset(Dataset):
    """A PACS dataset restricted to a fixed list of global indices (e.g. one
    domain's train or val split). Returns (PIL.Image RGB, int label)."""

    def __init__(self, hf_dataset, indices: list[int]):
        self.hf_dataset = hf_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        item = self.hf_dataset[self.indices[i]]
        img = item["image"].convert("RGB")
        return img, int(item["label"])

    @property
    def class_names(self) -> list[str]:
        return self.hf_dataset.features["label"].names
