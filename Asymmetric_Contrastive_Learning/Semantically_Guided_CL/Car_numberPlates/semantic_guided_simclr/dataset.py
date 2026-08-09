"""Dataset for semantic-guided training on generated CCPD real and manipulated images."""

from __future__ import annotations

from pathlib import Path

from common.augmentations import build_simclr_transform
from common.ccpd_data import CCPDGeneratedContrastiveDataset


def build_dataset(
    data_root: Path,
    image_size: int,
    train_fraction: float = 0.9,
    split_seed: int = 42,
    bbox_area_scale: float = 1.8,
) -> CCPDGeneratedContrastiveDataset:
    transform = build_simclr_transform(image_size)
    patch_transform = build_simclr_transform(image_size)
    return CCPDGeneratedContrastiveDataset(
        data_root,
        transform,
        split="train",
        train_fraction=train_fraction,
        split_seed=split_seed,
        include_plate_patch=True,
        patch_transform=patch_transform,
        bbox_area_scale=bbox_area_scale,
    )
