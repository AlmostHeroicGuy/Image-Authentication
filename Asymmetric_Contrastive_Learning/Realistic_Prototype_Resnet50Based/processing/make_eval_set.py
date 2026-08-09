"""
Create the held-out evaluation set used by eval.py.

The source test images may have arbitrary aspect ratios. Each image is first
preprocessed exactly like training/evaluation input: resize the shorter side to
the target size, then center-crop to a square. Positive and negative views are
generated from that square image so the saved triplets are already aligned.

Expected output layout:

    eval_set/
        real/
        positive/
        negative/
        masks/

If validation_set/validation_indices.npy or validation_filenames.txt exists,
those source images are excluded by default so the held-out evaluation set stays
separate from the validation threshold-selection set.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from augmentations import GlobalBenignAugmentation, LocalWatermarkForgery
from processing.preprocessing import preprocess_image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build eval_set triplets.")
    parser.add_argument("--test-root", type=Path, default=Path(r"C:\Users\tusha\Deep\Dataset\imagenet-100\test"))
    parser.add_argument("--out-root", type=Path, default=Path("eval_set"))
    parser.add_argument("--image-size", type=int, default=224, choices=[224, 256])
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument(
        "--exclude-indices",
        type=Path,
        default=Path("validation_set/validation_indices.npy"),
        help="Optional .npy indices to exclude from the source test set.",
    )
    parser.add_argument(
        "--exclude-filenames",
        type=Path,
        default=Path("validation_set/validation_filenames.txt"),
        help="Optional filename list to exclude from the source test set.",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Do not exclude validation indices even if the file exists.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove old files inside eval_set subfolders before writing.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(root):
    files = [
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def prepare_output_dirs(out_root, clean=False):
    for name in ("real", "positive", "negative", "masks"):
        folder = out_root / name
        if clean and folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def load_preprocessed_image(path, image_size):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return preprocess_image(img, image_size)


def filter_validation_images(
    files,
    exclude_indices_path,
    exclude_filenames_path,
    include_validation,
):
    if include_validation:
        return files, 0

    excluded_indices = set()
    excluded_filenames = set()

    if exclude_indices_path.exists():
        indices = np.load(exclude_indices_path)
        excluded_indices = {int(index) for index in indices.tolist()}

    if exclude_filenames_path.exists():
        with open(exclude_filenames_path, "r", encoding="utf-8") as handle:
            excluded_filenames = {
                line.strip() for line in handle
                if line.strip()
            }

    kept = []
    excluded_count = 0
    for index, path in enumerate(files):
        should_exclude = (
            index in excluded_indices
            or path.name in excluded_filenames
        )
        if should_exclude:
            excluded_count += 1
        else:
            kept.append(path)

    return kept, excluded_count


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_output_dirs(args.out_root, clean=args.clean)

    files = list_images(args.test_root)
    if not files:
        raise FileNotFoundError(f"No images found in {args.test_root}")

    source_count = len(files)
    files, excluded_count = filter_validation_images(
        files,
        args.exclude_indices,
        args.exclude_filenames,
        args.include_validation,
    )

    if excluded_count:
        print(f"Excluded {excluded_count} validation images from eval_set")

    if not files:
        raise RuntimeError("No images left for eval_set after exclusions.")

    global_aug = GlobalBenignAugmentation().to(device).eval()
    watermark_aug = LocalWatermarkForgery().to(device).eval()

    print(f"Found {source_count} source images in {args.test_root}")
    print(f"Generating {len(files)} held-out eval triplets at {args.image_size}x{args.image_size}")

    with torch.inference_mode():
        for index, img_path in enumerate(files):
            if index % 500 == 0:
                print(f"{index}/{len(files)}")

            x = load_preprocessed_image(img_path, args.image_size).unsqueeze(0).to(device)
            positive = global_aug(x)
            negative, soft_mask = watermark_aug(x)

            filename = f"{index:05d}.png"
            TF.to_pil_image(x.squeeze(0).cpu()).save(args.out_root / "real" / filename)
            TF.to_pil_image(positive.squeeze(0).clamp(0, 1).cpu()).save(
                args.out_root / "positive" / filename
            )
            TF.to_pil_image(negative.squeeze(0).clamp(0, 1).cpu()).save(
                args.out_root / "negative" / filename
            )
            TF.to_pil_image(soft_mask.squeeze(0).cpu()).save(
                args.out_root / "masks" / filename
            )

    print(f"{len(files)}/{len(files)}")
    print()
    print("=" * 60)
    print("Evaluation set created successfully.")
    print(f"Saved to: {args.out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
