"""Create visual sanity-check panels for generated CCPD swaps."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from .ccpd import parse_ccpd_filename
from .plate_ops import extract_plate, make_visualization_grid, read_image, write_image


def create_visualizations(
    dataset_root: Path,
    generated_root: Path,
    output_dir: Path,
    num_samples: int,
    seed: int,
) -> None:
    mapping_path = generated_root / "plate_mapping.json"
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    entries = payload["mapping"]
    rng = random.Random(seed)
    samples = rng.sample(entries, k=min(num_samples, len(entries)))
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(tqdm(samples, desc="Visualizing", unit="sample")):
        destination_rel = Path(entry["destination_path"])
        source_rel = Path(entry["source_path"])
        subset = destination_rel.parts[0]
        destination_inside_subset = Path(*destination_rel.parts[1:])
        original_path = dataset_root / destination_rel
        source_path = dataset_root / source_rel
        manipulated_path = generated_root / subset / "manipulated" / destination_inside_subset
        mask_path = generated_root / subset / "mask" / destination_inside_subset.with_suffix(".png")

        original = read_image(original_path)
        source = read_image(source_path)
        manipulated = read_image(manipulated_path)
        mask = read_image(mask_path)
        source_annotation = parse_ccpd_filename(source_path)
        plate = extract_plate(source, source_annotation.vertices)
        grid = make_visualization_grid(original, plate, mask[:, :, 0], manipulated)
        write_image(output_dir / f"sample_{i:04d}.jpg", grid)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Original CCPD root.")
    parser.add_argument("--generated-root", required=True, type=Path, help="Generated output root.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where visualization panels are written.")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    create_visualizations(
        dataset_root=args.dataset_root.expanduser().resolve(),
        generated_root=args.generated_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
