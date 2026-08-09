"""
Create the fixed 200-image visualization subset used by eval.py.

Images are preprocessed with the same resize-shorter-side + center-crop rule as
training/evaluation before augmentations are applied. This keeps the visual
heatmaps and saved triplets compatible with the 224/256 ResNet50 pipeline.
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
    parser = argparse.ArgumentParser(description="Build visualization_set triplets.")
    parser.add_argument("--test-root", type=Path, default=Path(r"C:\Users\tusha\Deep\Dataset\imagenet-100\test"))
    parser.add_argument("--out-root", type=Path, default=Path("visualization_set"))
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=224, choices=[224, 256])
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument(
        "--exclude-indices",
        type=Path,
        default=Path("validation_set/validation_indices.npy"),
        help="Optional .npy source indices to avoid sampling validation images.",
    )
    parser.add_argument(
        "--exclude-filenames",
        type=Path,
        default=Path("validation_set/validation_filenames.txt"),
        help="Optional filename list to avoid sampling validation images.",
    )
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="Allow visualization images to overlap validation images.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove old files inside visualization_set subfolders before writing.",
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
    for name in ("real", "positive", "negative"):
        folder = out_root / name
        if clean and folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def load_preprocessed_image(path, image_size):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return preprocess_image(img, image_size)


def available_source_indices(
    files,
    exclude_indices_path,
    exclude_filenames_path,
    include_validation,
):
    all_indices = np.arange(len(files))
    if include_validation:
        return all_indices

    excluded_indices = set()
    excluded_filenames = set()

    if exclude_indices_path.exists():
        excluded_indices = set(np.load(exclude_indices_path).astype(int).tolist())

    if exclude_filenames_path.exists():
        with open(exclude_filenames_path, "r", encoding="utf-8") as handle:
            excluded_filenames = {
                line.strip() for line in handle
                if line.strip()
            }

    return np.array(
        [
            index for index in all_indices
            if int(index) not in excluded_indices
            and files[int(index)].name not in excluded_filenames
        ],
        dtype=np.int64,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_output_dirs(args.out_root, clean=args.clean)

    files = list_images(args.test_root)
    if not files:
        raise FileNotFoundError(f"No images found in {args.test_root}")

    candidates = available_source_indices(
        files,
        args.exclude_indices,
        args.exclude_filenames,
        args.include_validation,
    )
    if len(candidates) < args.num_images:
        raise RuntimeError(
            f"Need {args.num_images} visualization images, but only "
            f"{len(candidates)} candidates are available."
        )

    selected_indices = np.random.choice(candidates, args.num_images, replace=False)
    selected_indices = np.sort(selected_indices)
    np.save(args.out_root / "visualization_indices.npy", selected_indices)

    global_aug = GlobalBenignAugmentation().to(device).eval()
    watermark_aug = LocalWatermarkForgery().to(device).eval()

    print(
        f"Generating {args.num_images} visualization triplets at "
        f"{args.image_size}x{args.image_size}"
    )

    with torch.inference_mode():
        for output_index, source_index in enumerate(selected_indices):
            if output_index % 20 == 0:
                print(f"{output_index}/{args.num_images}")

            x = load_preprocessed_image(files[int(source_index)], args.image_size)
            x = x.unsqueeze(0).to(device)
            positive = global_aug(x)
            negative, _ = watermark_aug(x)

            filename = f"{output_index:03d}.png"
            TF.to_pil_image(x.squeeze(0).cpu()).save(args.out_root / "real" / filename)
            TF.to_pil_image(positive.squeeze(0).clamp(0, 1).cpu()).save(
                args.out_root / "positive" / filename
            )
            TF.to_pil_image(negative.squeeze(0).clamp(0, 1).cpu()).save(
                args.out_root / "negative" / filename
            )

    print(f"{args.num_images}/{args.num_images}")
    print()
    print("=" * 60)
    print("Visualization set created successfully.")
    print(f"Saved to: {args.out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
