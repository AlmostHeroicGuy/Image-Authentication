"""
Generate a reproducible evaluation set with all 5000 test images.

Saves:
  eval_set/real/        - original images
  eval_set/positive/    - global benign augmented
  eval_set/negative/    - watermark forged
  eval_set/masks/       - soft masks from watermark forgery

All named with original filenames (00000.png, 00001.png, ..., 04999.png).
"""

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from augmentations import (
    GlobalBenignAugmentation,
    LocalWatermarkForgery
)

# ============================================================
# CONFIG
# ============================================================

TEST_ROOT = Path("TinyImageNet/test")
OUT_ROOT = Path("eval_set")

SEED = 69
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# OUTPUT FOLDERS
# ============================================================

(OUT_ROOT / "real").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "positive").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "negative").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "masks").mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD TEST IMAGES
# ============================================================

files = sorted(TEST_ROOT.glob("*.png"))

total = len(files)
print(f"Found {total} test images in {TEST_ROOT}")

# ============================================================
# AUGMENTATIONS
# ============================================================

global_aug = GlobalBenignAugmentation().to(DEVICE)
watermark_aug = LocalWatermarkForgery().to(DEVICE)

# ============================================================
# GENERATE EVALUATION SET
# ============================================================

print(f"Generating evaluation set with {total} triplets + masks...")

for idx, img_path in enumerate(files):

    if idx % 500 == 0:
        print(f"{idx}/{total}")

    img = Image.open(img_path).convert("RGB")
    x = TF.to_tensor(img).unsqueeze(0).to(DEVICE)  # (1,3,64,64)

    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    positive = global_aug(x)

    # --------------------------------------------------------
    # Negative & Mask
    # --------------------------------------------------------

    negative, soft_mask = watermark_aug(x)

    # --------------------------------------------------------
    # Save with original filename
    # --------------------------------------------------------

    filename = f"{idx:05d}.png"

    TF.to_pil_image(
        x.squeeze(0).cpu()
    ).save(
        OUT_ROOT / "real" / filename
    )

    TF.to_pil_image(
        positive.squeeze(0).clamp(0, 1).cpu()
    ).save(
        OUT_ROOT / "positive" / filename
    )

    TF.to_pil_image(
        negative.squeeze(0).clamp(0, 1).cpu()
    ).save(
        OUT_ROOT / "negative" / filename
    )

    # Save the soft mask (B,1,H,W) -> (H,W) as grayscale
    TF.to_pil_image(
        soft_mask.squeeze(0).cpu()
    ).save(
        OUT_ROOT / "masks" / filename
    )

print()
print("=" * 60)
print("Evaluation set created successfully.")
print(f"Saved to: {OUT_ROOT}")
print("=" * 60)
print()
print("Structure:")
print(f"  {OUT_ROOT}/real/")
print(f"  {OUT_ROOT}/positive/")
print(f"  {OUT_ROOT}/negative/")
print(f"  {OUT_ROOT}/masks/")
print()
