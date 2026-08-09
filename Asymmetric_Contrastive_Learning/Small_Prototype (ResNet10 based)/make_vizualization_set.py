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

OUT_ROOT = Path("visualization_set")

NUM_IMAGES = 200
SEED = 69

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# OUTPUT FOLDERS
# ============================================================

(OUT_ROOT / "real").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "positive").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "negative").mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD TEST IMAGES
# ============================================================

files = sorted(TEST_ROOT.glob("*.png"))

assert len(files) >= NUM_IMAGES, (
    f"Found only {len(files)} test images."
)

# ============================================================
# FIXED SUBSET
# ============================================================

indices = np.random.choice(
    len(files),
    NUM_IMAGES,
    replace=False
)

indices = np.sort(indices)

np.save(
    OUT_ROOT / "visualization_indices.npy",
    indices
)

# ============================================================
# AUGMENTATIONS
# ============================================================

global_aug = GlobalBenignAugmentation()
watermark_aug = LocalWatermarkForgery()

# ============================================================
# GENERATE VISUALIZATION SET
# ============================================================

print(f"Generating {NUM_IMAGES} visualization triplets...")

for k, idx in enumerate(indices):

    if k % 20 == 0:
        print(f"{k}/{NUM_IMAGES}")

    img_path = files[idx]

    img = Image.open(img_path).convert("RGB")

    x = TF.to_tensor(img).unsqueeze(0)   # (1,3,64,64)

    # --------------------------------------------------------
    # Positive
    # --------------------------------------------------------

    positive = global_aug(x)

    # --------------------------------------------------------
    # Negative
    # --------------------------------------------------------

    negative, _ = watermark_aug(x)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    TF.to_pil_image(
        x.squeeze(0)
    ).save(
        OUT_ROOT / "real" / f"{k:03d}.png"
    )

    TF.to_pil_image(
        positive.squeeze(0).clamp(0, 1)
    ).save(
        OUT_ROOT / "positive" / f"{k:03d}.png"
    )

    TF.to_pil_image(
        negative.squeeze(0).clamp(0, 1)
    ).save(
        OUT_ROOT / "negative" / f"{k:03d}.png"
    )

print()
print("=" * 60)
print("Visualization set created successfully.")
print(f"Saved to: {OUT_ROOT}")
print("=" * 60)