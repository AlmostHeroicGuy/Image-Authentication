# Dataset Generation for Image Authentication Evaluation

## Overview

This script generates a **controlled, reproducible dataset** for evaluating the Lin & Chang (2001) image authentication algorithm. Images are automatically classified into three categories based on their index number.

## Classification Rules

After automatic renaming to `img1.jpg, img2.jpg, img3.jpg, ...`, each image is processed according to its index **N**:

### 1. **Prime Indices (N ∈ PRIMES)**
- **Type**: JPEG Compressed
- **Compression Quality**: 80
- **Purpose**: Test if authentication tolerates benign lossy compression
- **Examples**: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, ...

### 2. **Multiples of 5 (N ∈ {10, 15, 20, 25, ...})**
- **Type**: Identical (unchanged)
- **Purpose**: Test true positive authentication (must pass)
- **Note**: Excludes 5 itself (which is prime)

### 3. **All Other Indices**
- **Type**: Tampered (copy-paste)
- **Tampering Method**: Random patch from another image
- **Patch Size**: 10-15% of image height × 10-15% of image width
- **Purpose**: Test detection of malicious manipulation

## File Structure

```
input_images/           # Place raw JPG files here
├── photo1.jpg
├── photo2.jpg
└── ...

generated/              # Output location (created automatically)
├── img1_processed.jpg  (Compressed)
├── img2_processed.jpg  (Identical)
├── img3_processed.jpg  (Compressed)
├── img4_processed.jpg  (Tampered)
└── ...
```

## Usage

```bash
python generate_dataset.py
```

**Output**:
```
[STEP 1] Renaming images to sequential IDs...
[OK] Renamed 10 images.

[STEP 2] Processing 10 images...
============================================================
[  1] COMPRESSED (prime)
[  2] COMPRESSED (prime)
[  3] COMPRESSED (prime)
[  4] TAMPERED (using img2.jpg)
[  5] COMPRESSED (prime)
[  6] TAMPERED (using img3.jpg)
[  7] COMPRESSED (prime)
[  8] TAMPERED (using img1.jpg)
[  9] TAMPERED (using img4.jpg)
[ 10] IDENTICAL (multiple of 5)
============================================================

[COMPLETE] Dataset generation finished!
  - Compressed images: 5
  - Identical images:  1
  - Tampered images:   4
  - Total processed:   10/10

Output folder: generated/
All processed images follow pattern: imgN_processed.jpg
```

## Why This Design?

### ✓ **Reproducibility**
- Entire classification depends only on index and precomputed prime sets
- Same input always produces same output
- Easy to regenerate from raw images

### ✓ **Balanced Evaluation**
Tests the three key capabilities of the authentication algorithm:
- Authenticates identical originals
- Tolerates JPEG compression
- Detects tampering/splicing

### ✓ **Realistic Tampering**
- Copy-paste from random images simulates real-world splicing
- More realistic than synthetic noise or uniform filtering
- Patch locations/sizes are randomized for variety

### ✓ **Scalable**
- Handles 10 or 10,000 images with identical logic
- Requires no manual intervention
- Index-based classification is O(1)

## Technical Details

### Prime Numbers (up to 300)
Used precomputed list to avoid expensive prime checking per image:
```python
PRIMES = (2, 3, 5, 7, 11, 13, ..., 293)
```

### Multiples of 5
Computed as: `{10, 15, 20, 25, ..., 300}`

### Tampering Algorithm
1. Extract patch from random donor image (10-15% of source area)
2. Ensure patch fits in donor and source images
3. Paste at random location in source image
4. Save tampered result

### Error Handling
- Bounds checking for patch extraction/pasting
- Fallback to smaller patch if needed
- Try/except blocks for robustness

## Next Steps

After dataset generation:

1. **Feature Extraction**: Run `image_auth.py` to extract DCT features from each image
2. **Evaluation**: Compute authentication accuracy, FPR, FNR
3. **Analysis**: Generate heatmaps and statistics of detected manipulations
4. **Report**: Document robustness to compression vs. tampering

## Mathematical Foundation

The classification is based on number theory:

- **Prime (N)**: Small, sparse, distributed irregularly
- **Multiple of 5**: Regular, predictable pattern
- **Other**: Residual category

This creates a mathematically clean partition of the dataset with no overlap.

---

**Dataset Format**: Suitable for evaluating DCT-based image authentication methods

**Citation**: Based on "A Robust Image Authentication Method Distinguishing JPEG Compression from Malicious Manipulation" by Ching-Yung Lin and Shih-Fu Chang (2001)
