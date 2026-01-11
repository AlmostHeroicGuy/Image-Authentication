import os
import random
import shutil
from PIL import Image
import numpy as np

# ============================================================
# PRECOMPUTED INDEX SETS (PRIMES & MULTIPLES OF 5)
# ============================================================

PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293
)

# Multiples of 5 excluding 5 itself
MULTIPLES_OF_5 = tuple(i for i in range(10, 301, 5))

# ============================================================
# RENAME FILES → img1.jpg, img2.jpg, img3.jpg ...
# ============================================================

def rename_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    
    for idx, fname in enumerate(files, start=1):
        new_name = f"img{idx}.jpg"
        os.rename(os.path.join(folder, fname), os.path.join(folder, new_name))
    
    print(f"[OK] Renamed {len(files)} images.")

# ============================================================
# JPEG COMPRESSION (Quality = 80)
# ============================================================

def compress_image(input_path, output_path):
    img = Image.open(input_path)
    img.save(output_path, "JPEG", quality=80)

# ============================================================
# TAMPERING — COPY PATCH FROM ANOTHER RANDOM IMAGE
# ============================================================

def tamper_image(src_path, donor_path, output_path):
    """
    Apply copy-paste tampering by extracting a patch from donor image 
    and pasting it into source image at random location.
    
    Args:
        src_path: Source image to tamper with
        donor_path: Donor image to extract patch from
        output_path: Output tampered image path
    """
    src = Image.open(src_path).convert("L")
    donor = Image.open(donor_path).convert("L")

    src_arr = np.array(src)
    donor_arr = np.array(donor)

    H_src, W_src = src_arr.shape
    H_donor, W_donor = donor_arr.shape

    # Patch size = 10–15% of source image area
    patch_h = random.randint(int(0.10 * H_src), int(0.15 * H_src))
    patch_w = random.randint(int(0.10 * W_src), int(0.15 * W_src))

    # Extract donor patch with bounds checking
    # If patch doesn't fit in donor, shrink it or retry
    max_retries = 5
    patch_extracted = False
    
    for attempt in range(max_retries):
        # Ensure patch fits in donor image
        if patch_h >= H_donor or patch_w >= W_donor:
            # Shrink patch size
            patch_h = min(patch_h, H_donor - 1)
            patch_w = min(patch_w, W_donor - 1)
        
        # Random coordinates in donor with bounds checking
        dx = random.randint(0, max(0, H_donor - patch_h - 1))
        dy = random.randint(0, max(0, W_donor - patch_w - 1))
        
        # Extract patch safely
        try:
            patch = donor_arr[dx:dx+patch_h, dy:dy+patch_w]
            if patch.shape == (patch_h, patch_w):
                patch_extracted = True
                break
        except:
            continue
    
    if not patch_extracted:
        # Fallback: use a smaller default patch
        patch_h = min(int(0.10 * H_src), H_donor - 1, H_src - 1)
        patch_w = min(int(0.10 * W_src), W_donor - 1, W_src - 1)
        dx = max(0, H_donor - patch_h - 1)
        dy = max(0, W_donor - patch_w - 1)
        patch = donor_arr[dx:dx+patch_h, dy:dy+patch_w]

    # Paste patch into source image with bounds checking
    px = random.randint(0, max(0, H_src - patch_h - 1))
    py = random.randint(0, max(0, W_src - patch_w - 1))

    # Ensure paste region fits
    paste_h = min(patch_h, H_src - px)
    paste_w = min(patch_w, W_src - py)
    patch_resized = patch[:paste_h, :paste_w]

    # Apply tampering
    src_arr[px:px+paste_h, py:py+paste_w] = patch_resized

    tampered = Image.fromarray(src_arr)
    tampered.save(output_path)

# ============================================================
# MAIN DATASET GENERATION FUNCTION
# ============================================================

def generate_dataset():
    """
    Generate a controlled dataset for image authentication evaluation.
    
    Classification by index:
      - Prime indices → JPEG Compressed (quality=80)
      - Multiples of 5 → Identical (unchanged)
      - All others → Tampered (copy-paste from random image)
    """
    input_folder = "input_images"
    output_folder = "generated"

    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Rename images to sequential IDs
    print("\n[STEP 1] Renaming images to sequential IDs...")
    rename_images(input_folder)

    # Get renamed images list
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
    
    if not files:
        print("[ERROR] No JPG files found in input_images/")
        return

    print(f"\n[STEP 2] Processing {len(files)} images...")
    print("=" * 60)

    compressed_count = 0
    identical_count = 0
    tampered_count = 0

    for fname in files:
        idx = int(fname[3:-4])   # extract N from "imgN.jpg"

        src_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, f"img{idx}_processed.jpg")

        try:
            # ----- PRIME → COMPRESSED -----
            if idx in PRIMES:
                compress_image(src_path, out_path)
                print(f"[{idx:3d}] COMPRESSED (prime)")
                compressed_count += 1

            # ----- MULTIPLE OF 5 → IDENTICAL -----
            elif idx in MULTIPLES_OF_5:
                shutil.copy(src_path, out_path)
                print(f"[{idx:3d}] IDENTICAL (multiple of 5)")
                identical_count += 1

            # ----- OTHERS → TAMPERED -----
            else:
                donor_choices = [x for x in files if x != fname]
                donor = random.choice(donor_choices)
                donor_path = os.path.join(input_folder, donor)

                tamper_image(src_path, donor_path, out_path)
                print(f"[{idx:3d}] TAMPERED (using {donor})")
                tampered_count += 1
        
        except Exception as e:
            print(f"[{idx:3d}] ERROR: {str(e)}")

    print("=" * 60)
    print(f"\n[COMPLETE] Dataset generation finished!")
    print(f"  - Compressed images: {compressed_count}")
    print(f"  - Identical images:  {identical_count}")
    print(f"  - Tampered images:   {tampered_count}")
    print(f"  - Total processed:   {compressed_count + identical_count + tampered_count}/{len(files)}")
    print(f"\nOutput folder: {output_folder}/")
    print(f"All processed images follow pattern: imgN_processed.jpg")

# ============================================================

if __name__ == "__main__":
    generate_dataset()
