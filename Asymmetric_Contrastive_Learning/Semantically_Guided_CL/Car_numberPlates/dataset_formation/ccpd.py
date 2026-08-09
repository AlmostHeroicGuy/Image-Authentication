"""CCPD filename parsing and dataset indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CCPDAnnotation:
    """Annotations encoded in one CCPD filename."""

    area_ratio: str
    tilt: tuple[float, float]
    bbox: np.ndarray
    vertices: np.ndarray
    plate_text: str
    brightness: str
    blur: str


@dataclass(frozen=True)
class ImageRecord:
    """One indexed image and its parsed CCPD annotations."""

    index: int
    path: Path
    subset: str
    relative_path: Path
    annotation: CCPDAnnotation


def parse_ccpd_filename(path: str | Path) -> CCPDAnnotation:
    """Parse CCPD annotations from an image filename.

    CCPD names follow:
    area-tilt-bbox-vertices-plate-brightness-blur.jpg
    """

    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Expected at least 7 CCPD fields, got {len(parts)}: {path}")

    tilt = _parse_float_pair(parts[1], "_")
    bbox = _parse_points(parts[2], "_")
    vertices = _parse_points(parts[3], "_")
    if bbox.shape != (2, 2):
        raise ValueError(f"Expected bbox with 2 points, got {bbox.shape}: {path}")
    if vertices.shape != (4, 2):
        raise ValueError(f"Expected 4 plate vertices, got {vertices.shape}: {path}")

    return CCPDAnnotation(
        area_ratio=parts[0],
        tilt=tilt,
        bbox=bbox.astype(np.float32),
        vertices=vertices.astype(np.float32),
        plate_text=parts[4],
        brightness=parts[5],
        blur=parts[6],
    )


def build_image_index(
    dataset_root: Path,
    subsets: Sequence[str],
    image_extensions: Iterable[str],
    limit: int | None = None,
) -> list[ImageRecord]:
    """Collect and parse images from the requested CCPD subsets."""

    extensions = {ext.lower() for ext in image_extensions}
    records: list[ImageRecord] = []
    next_index = 0

    for subset in subsets:
        subset_root = dataset_root / subset
        if not subset_root.exists():
            continue
        for path in sorted(subset_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            annotation = parse_ccpd_filename(path)
            records.append(
                ImageRecord(
                    index=next_index,
                    path=path,
                    subset=subset,
                    relative_path=Path(subset) / path.relative_to(subset_root),
                    annotation=annotation,
                )
            )
            next_index += 1
            if limit is not None and len(records) >= limit:
                return records

    return records


def _parse_float_pair(value: str, delimiter: str) -> tuple[float, float]:
    parts = value.split(delimiter)
    if len(parts) != 2:
        raise ValueError(f"Expected two values in '{value}'")
    return float(parts[0]), float(parts[1])


def _parse_points(value: str, point_delimiter: str) -> np.ndarray:
    points = []
    for point in value.split(point_delimiter):
        xy = point.split("&")
        if len(xy) != 2:
            raise ValueError(f"Expected x&y point in '{point}'")
        points.append((float(xy[0]), float(xy[1])))
    return np.asarray(points, dtype=np.float32)

