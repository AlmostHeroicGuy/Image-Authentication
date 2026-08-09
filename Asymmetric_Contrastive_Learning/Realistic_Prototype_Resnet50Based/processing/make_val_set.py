"""
Create the validation set used only for threshold selection in eval.py.

This script expects a folder that already contains the selected validation
images. It uses every image in that folder and saves aligned
real/positive/negative triplets:

    validation_set/
        real/
        positive/
        negative/

It also writes validation_filenames.txt so the exact source images are recorded.
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Build validation_set triplets.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"C:\Users\tusha\Deep\Dataset\imagenet-100\validation"),
        help="Folder containing the already selected validation images.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("validation_set"))
    parser.add_argument("--image-size", type=int, default=224, choices=[224, 256])
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove old files inside validation_set subfolders before writing.",
    )
    return parser.parse_args()


def set_seed(seed):
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


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_output_dirs(args.out_root, clean=args.clean)

    files = list_images(args.source_root)
    if not files:
        raise FileNotFoundError(f"No images found in {args.source_root}")

    num_images = len(files)

    with open(args.out_root / "validation_filenames.txt", "w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{path.name}\n")

    global_aug = GlobalBenignAugmentation().to(device).eval()
    watermark_aug = LocalWatermarkForgery().to(device).eval()

    print(
        f"Generating {num_images} validation triplets at "
        f"{args.image_size}x{args.image_size}"
    )

    with torch.inference_mode():
        for output_index, img_path in enumerate(files):
            if output_index % 250 == 0:
                print(f"{output_index}/{num_images}")

            x = load_preprocessed_image(img_path, args.image_size)
            x = x.unsqueeze(0).to(device)
            positive = global_aug(x)
            negative, _ = watermark_aug(x)

            filename = f"{output_index:05d}.png"
            TF.to_pil_image(x.squeeze(0).cpu()).save(args.out_root / "real" / filename)
            TF.to_pil_image(positive.squeeze(0).clamp(0, 1).cpu()).save(
                args.out_root / "positive" / filename
            )
            TF.to_pil_image(negative.squeeze(0).clamp(0, 1).cpu()).save(
                args.out_root / "negative" / filename
            )

    print(f"{num_images}/{num_images}")
    print()
    print("=" * 60)
    print("Validation set created successfully.")
    print(f"Saved to: {args.out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
