"""Image loading helpers for the HDA-HQSwap train/original split."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    """Recursively list image files under ``root``."""

    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise FileNotFoundError(f"No images found under: {root}")
    return sorted(paths)


class SimCLRImageDataset(Dataset):
    """Return two independently augmented views of each original image."""

    def __init__(
        self,
        image_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        return_pil: bool = False,
    ) -> None:
        self.image_paths = list_images(image_dir)
        self.transform = transform
        self.return_pil = return_pil

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            view_1 = self.transform(image)
            view_2 = self.transform(image)
            if self.return_pil:
                return view_1, view_2, image.copy(), str(path)
            return view_1, view_2, str(path)


def collate_with_pil(batch: Iterable[tuple[torch.Tensor, torch.Tensor, Image.Image, str]]):
    """Collate tensors while leaving PIL images in a Python list."""

    views_1, views_2, images, paths = zip(*batch)
    return torch.stack(list(views_1)), torch.stack(list(views_2)), list(images), list(paths)

