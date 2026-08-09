"""CLI for generating a manipulated CCPD plate-swapping dataset."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from .ccpd import ImageRecord, build_image_index
from .config import DEFAULT_CCPD_SUBSETS, GenerationConfig
from .mapping import build_derangement, save_mapping
from .plate_ops import extract_plate, insert_plate, read_image, write_image


LOGGER = logging.getLogger("dataset_formation.generate")


@dataclass(frozen=True)
class ProcessResult:
    destination: str
    ok: bool
    skipped: bool = False
    error_stage: str | None = None
    error: str | None = None


def generate_dataset(config: GenerationConfig) -> dict[str, float | int]:
    """Generate real/manipulated/mask outputs and return processing statistics."""

    start = time.perf_counter()
    records = build_image_index(
        config.dataset_root,
        config.subsets,
        config.image_extensions,
        config.limit,
    )
    if len(records) < 2:
        raise ValueError("Need at least two valid CCPD images to generate swaps")

    permutation = build_derangement(records, config.seed)
    save_mapping(config.output_root / "plate_mapping.json", records, permutation, config.seed)
    _prepare_output_dirs(config.output_root, config.subsets)

    jobs = [
        (
            record,
            records[source_idx],
            config.output_root,
            config.overwrite,
            config.jpeg_quality,
        )
        for record, source_idx in zip(records, permutation)
    ]

    processed = 0
    skipped = 0
    failures: dict[str, int] = {}

    if config.workers == 1:
        iterator = (_process_one(*job) for job in jobs)
        progress = tqdm(iterator, total=len(jobs), desc="Generating", unit="image")
        for result in progress:
            processed, skipped = _update_stats(result, failures, processed, skipped)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(_process_one, *job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating", unit="image"):
                result = future.result()
                processed, skipped = _update_stats(result, failures, processed, skipped)

    elapsed = time.perf_counter() - start
    speed = processed / elapsed if elapsed > 0 else 0.0
    LOGGER.info("Processed images: %s", processed)
    LOGGER.info("Skipped images: %s", skipped)
    LOGGER.info("Failures by stage: %s", failures)
    LOGGER.info("Average processing speed: %.2f images/s", speed)
    return {
        "processed": processed,
        "skipped": skipped,
        "failures": sum(failures.values()),
        "elapsed_seconds": elapsed,
        "images_per_second": speed,
    }


def _process_one(
    destination_record: ImageRecord,
    source_record: ImageRecord,
    output_root: Path,
    overwrite: bool,
    jpeg_quality: int,
) -> ProcessResult:
    destination_rel = destination_record.relative_path
    real_path = output_root / destination_record.subset / "real" / destination_rel.relative_to(destination_record.subset)
    manipulated_path = output_root / destination_record.subset / "manipulated" / destination_rel.relative_to(destination_record.subset)
    mask_path = output_root / destination_record.subset / "mask" / destination_rel.with_suffix(".png").relative_to(destination_record.subset)

    if not overwrite and real_path.exists() and manipulated_path.exists() and mask_path.exists():
        return ProcessResult(destination=destination_rel.as_posix(), ok=True, skipped=True)

    try:
        real_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination_record.path, real_path)
    except Exception as exc:  # pragma: no cover - exercised in full dataset runs
        return ProcessResult(destination=destination_rel.as_posix(), ok=False, error_stage="copy", error=str(exc))

    try:
        destination_image = read_image(destination_record.path)
        source_image = read_image(source_record.path)
        source_plate = extract_plate(source_image, source_record.annotation.vertices)
    except Exception as exc:  # pragma: no cover
        return ProcessResult(destination=destination_rel.as_posix(), ok=False, error_stage="extraction", error=str(exc))

    try:
        manipulated, mask = insert_plate(
            destination_image,
            destination_record.annotation.vertices,
            source_plate,
        )
    except Exception as exc:  # pragma: no cover
        return ProcessResult(destination=destination_rel.as_posix(), ok=False, error_stage="insertion", error=str(exc))

    try:
        write_image(manipulated_path, manipulated, jpeg_quality=jpeg_quality)
        write_image(mask_path, mask, jpeg_quality=jpeg_quality)
    except Exception as exc:  # pragma: no cover
        return ProcessResult(destination=destination_rel.as_posix(), ok=False, error_stage="write", error=str(exc))

    return ProcessResult(destination=destination_rel.as_posix(), ok=True)


def _prepare_output_dirs(output_root: Path, subsets: Sequence[str]) -> None:
    for subset in subsets:
        for split in ("real", "manipulated", "mask"):
            (output_root / subset / split).mkdir(parents=True, exist_ok=True)


def _update_stats(
    result: ProcessResult,
    failures: dict[str, int],
    processed: int,
    skipped: int,
) -> tuple[int, int]:
    if result.skipped:
        skipped += 1
    elif result.ok:
        processed += 1
    else:
        stage = result.error_stage or "unknown"
        failures[stage] = failures.get(stage, 0) + 1
        LOGGER.warning("Failed %s during %s: %s", result.destination, stage, result.error)
    return processed, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Root containing CCPD subset folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Separate root for generated dataset.")
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_CCPD_SUBSETS), help="CCPD subsets to process.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic mapping seed.")
    parser.add_argument("--workers", type=int, default=1, help="CPU worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for local testing.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for manipulated images.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    config = GenerationConfig.from_paths(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        subsets=tuple(args.subsets),
        seed=args.seed,
        workers=max(1, args.workers),
        limit=args.limit,
        overwrite=args.overwrite,
        jpeg_quality=args.jpeg_quality,
    )
    generate_dataset(config)


if __name__ == "__main__":
    main()

