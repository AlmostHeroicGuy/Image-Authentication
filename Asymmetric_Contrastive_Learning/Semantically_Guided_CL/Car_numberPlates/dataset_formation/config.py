"""Configuration objects for CCPD plate swapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_CCPD_SUBSETS: tuple[str, ...] = (
    "ccpd_base",
    "ccpd_blur",
    "ccpd_challenge",
    "ccpd_db",
    "ccpd_fn",
    "ccpd_rotate",
    "ccpd_tilt",
    "ccpd_weather",
)


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime configuration for manipulated dataset generation."""

    dataset_root: Path
    output_root: Path
    subsets: Sequence[str] = DEFAULT_CCPD_SUBSETS
    seed: int = 1337
    workers: int = 1
    limit: int | None = None
    overwrite: bool = False
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    jpeg_quality: int = 95

    @classmethod
    def from_paths(
        cls,
        dataset_root: str | Path,
        output_root: str | Path,
        **kwargs: object,
    ) -> "GenerationConfig":
        return cls(
            dataset_root=Path(dataset_root).expanduser().resolve(),
            output_root=Path(output_root).expanduser().resolve(),
            **kwargs,
        )

