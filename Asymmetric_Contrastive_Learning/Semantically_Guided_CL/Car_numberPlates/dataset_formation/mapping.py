"""Deterministic source-to-destination plate assignment."""

from __future__ import annotations

import json
import random
from pathlib import Path

from .ccpd import ImageRecord


def build_derangement(records: list[ImageRecord], seed: int) -> list[int]:
    """Create a deterministic permutation with no image mapped to itself."""

    n = len(records)
    if n < 2:
        raise ValueError("At least two images are required to create plate swaps")

    rng = random.Random(seed)
    permutation = list(range(n))
    rng.shuffle(permutation)

    while True:
        conflicts = [i for i, source_idx in enumerate(permutation) if source_idx == i]
        if not conflicts:
            return permutation
        for i in conflicts:
            candidates = [j for j in range(n) if j != i and permutation[j] != i]
            if not candidates:
                rng.shuffle(permutation)
                break
            j = rng.choice(candidates)
            permutation[i], permutation[j] = permutation[j], permutation[i]


def save_mapping(path: Path, records: list[ImageRecord], permutation: list[int], seed: int) -> None:
    """Save the assignment in a reproducible, inspectable JSON format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "num_images": len(records),
        "mapping": [
            {
                "destination_index": record.index,
                "destination_path": record.relative_path.as_posix(),
                "source_index": records[source_idx].index,
                "source_path": records[source_idx].relative_path.as_posix(),
            }
            for record, source_idx in zip(records, permutation)
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

