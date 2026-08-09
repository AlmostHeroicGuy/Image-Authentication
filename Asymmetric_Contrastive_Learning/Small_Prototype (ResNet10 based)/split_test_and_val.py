import os
import random
import shutil

TEST_DIR = "TinyImageNet/test"
VAL_DIR = "TinyImageNet/val"

os.makedirs(VAL_DIR, exist_ok=True)

# Get all PNG files
files = [f for f in os.listdir(TEST_DIR) if f.endswith(".png")]

print(f"Found {len(files)} test images.")

# Safety check
assert len(files) == 10000, f"Expected 10000 images, found {len(files)}"

# Randomly select 5000 for validation
random.seed(69)   # reproducible split
val_files = set(random.sample(files, 5000))

# Move them to val/
for fname in val_files:
    shutil.move(
        os.path.join(TEST_DIR, fname),
        os.path.join(VAL_DIR, fname)
    )

print("Done!")
print(f"Validation images: {len(os.listdir(VAL_DIR))}")
print(f"Remaining test images: {len(os.listdir(TEST_DIR))}")