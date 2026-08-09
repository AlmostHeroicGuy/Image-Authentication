"""CCPD generated-dataset loading and annotation-derived plate crops."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_formation.ccpd import parse_ccpd_filename


CCPD_SUBSETS: tuple[str, ...] = (
    "ccpd_base",
    "ccpd_blur",
    "ccpd_challenge",
    "ccpd_db",
    "ccpd_fn",
    "ccpd_rotate",
    "ccpd_tilt",
    "ccpd_weather",
)

VARIANTS: tuple[str, ...] = ("real", "manipulated")
SplitName = Literal["train", "test", "all"]


@dataclass(frozen=True)
class CCPDGeneratedSample:
    """One real or manipulated image from the generated CCPD dataset."""

    path: Path
    subset: str
    variant: str
    pair_key: str


def list_generated_ccpd_samples(root: Path, subsets: tuple[str, ...] = CCPD_SUBSETS) -> list[CCPDGeneratedSample]:
    """List real and manipulated images while preserving paired sample keys."""

    samples: list[CCPDGeneratedSample] = []
    for subset in subsets:
        for variant in VARIANTS:
            variant_root = root / subset / variant
            if not variant_root.exists():
                continue
            for path in sorted(variant_root.rglob("*")):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                relative_inside_variant = path.relative_to(variant_root)
                samples.append(
                    CCPDGeneratedSample(
                        path=path,
                        subset=subset,
                        variant=variant,
                        pair_key=(Path(subset) / relative_inside_variant).as_posix(),
                    )
                )

    if not samples:
        raise FileNotFoundError(f"No generated CCPD real/manipulated images found under: {root}")
    return samples


def split_generated_ccpd_samples(
    samples: list[CCPDGeneratedSample],
    train_fraction: float,
    seed: int,
    split: SplitName,
) -> list[CCPDGeneratedSample]:
    """Split by pair key so corresponding real/manipulated images stay together."""

    if split == "all":
        return samples
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")

    pair_keys = sorted({sample.pair_key for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(pair_keys)
    train_count = max(1, min(len(pair_keys) - 1, int(round(len(pair_keys) * train_fraction))))
    selected_keys = set(pair_keys[:train_count] if split == "train" else pair_keys[train_count:])
    return [sample for sample in samples if sample.pair_key in selected_keys]


class CCPDGeneratedContrastiveDataset(Dataset):
    """Return global SimCLR views and optional plate-context local views."""

    def __init__(
        self,
        data_root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        split: SplitName = "train",
        train_fraction: float = 0.9,
        split_seed: int = 42,
        include_plate_patch: bool = False,
        patch_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        bbox_area_scale: float = 1.8,
        subsets: tuple[str, ...] = CCPD_SUBSETS,
    ) -> None:
        samples = list_generated_ccpd_samples(data_root, subsets)
        self.samples = split_generated_ccpd_samples(samples, train_fraction, split_seed, split)
        if not self.samples:
            raise FileNotFoundError(f"No samples found for split={split} under: {data_root}")
        self.transform = transform
        self.include_plate_patch = include_plate_patch
        self.patch_transform = patch_transform or transform
        self.bbox_area_scale = bbox_area_scale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            view_1 = self.transform(image)
            view_2 = self.transform(image)
            if not self.include_plate_patch:
                return view_1, view_2, str(sample.path)

            patch = crop_expanded_ccpd_bbox(image, sample.path, self.bbox_area_scale)
            patch_view_1 = self.patch_transform(patch)
            patch_view_2 = self.patch_transform(patch)
            return view_1, view_2, patch_view_1, patch_view_2, str(sample.path)


def crop_expanded_ccpd_bbox(image: Image.Image, image_path: Path, area_scale: float = 1.8) -> Image.Image:
    """Crop the CCPD bbox expanded by area_scale while preserving aspect ratio."""

    if area_scale <= 0:
        raise ValueError(f"area_scale must be positive, got {area_scale}")
    annotation = parse_ccpd_filename(image_path)
    x1 = float(annotation.bbox[:, 0].min())
    y1 = float(annotation.bbox[:, 1].min())
    x2 = float(annotation.bbox[:, 0].max())
    y2 = float(annotation.bbox[:, 1].max())
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    side_scale = math.sqrt(area_scale)
    expanded_width = width * side_scale
    expanded_height = height * side_scale
    crop_box = (
        max(0, int(round(center_x - expanded_width / 2.0))),
        max(0, int(round(center_y - expanded_height / 2.0))),
        min(image.width, int(round(center_x + expanded_width / 2.0))),
        min(image.height, int(round(center_y + expanded_height / 2.0))),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError(f"Expanded CCPD bbox is empty for: {image_path}")
    return image.crop(crop_box).convert("RGB")

