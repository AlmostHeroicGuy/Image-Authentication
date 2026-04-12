# Image Authentication Research Project

A research project on **semi-fragile image authentication** — the problem of verifying that an image has not been maliciously manipulated, while still tolerating a broad class of legitimate, content-preserving operations like JPEG compression, filtering, scaling, and enhancement.

---

## The Core Problem

Traditional image authentication is binary: either every pixel matches, or the image is rejected. This is too strict for the real world. Consider a photograph transmitted over the internet — it may be JPEG-compressed, re-encoded, slightly scaled, or brightness-adjusted along the way. None of these operations change the *visual meaning* or *content* of the image. A purely hash-based authenticator would reject all of them.

The actual goal of semi-fragile image authentication is to make a distinction based on *intent and effect*, not just pixel-level equality:

**Acceptable manipulations** (should pass authentication):
- JPEG lossy compression at any quality factor or compression ratio
- Multiple rounds of recompression
- Integer rounding errors from DCT/IDCT operations
- Image filtering (low-pass, sharpening, enhancement)
- Constant intensity changes (brightness adjustment)
- Scaling and resampling (with resize-back to original)
- Format conversion and lossless compression
- Color space transformations

**Malicious manipulations** (must be detected):
- Content replacement — copying pixels from another region or image and pasting them in
- Splicing — compositing parts of multiple images
- Cloning — duplicating parts of the same image to hide or add content
- Any modification that changes the visual meaning of the image to an observer

The fundamental difficulty is that the authenticator cannot know the *purpose* of a manipulation — only its *method and effect*. The solution this project builds on (Lin & Chang, 2001) sidesteps this by identifying a mathematical invariant that *acceptable* manipulations preserve but *malicious* ones typically do not: the sign of DCT coefficient differences between image block pairs.

This repo implements and extends that classical approach, while also exploring two related directions: algorithm unrolling (LISTA) as a framework for understanding learned sparse representations, and contrastive learning (SimCLR) as a way to learn image representations that could generalize deepfake or tamper detection beyond the classical scheme.

---

## Repository Structure

```
Image-Authentication/
│
├── Lin&Chang implementation/          # Classical semi-fragile authentication
│   ├── image_auth.py                  # Core ImageAuthenticator class
│   ├── verification_casia.py          # Evaluation on CASIA tampered image dataset
│   ├── verification_casia_gt.py       # Ground-truth-based CASIA evaluation
│   ├── casia_mapping.py               # CASIA filename utilities
│   ├── extract_samples.py             # Sample extraction helper
│   ├── README.md                      # Implementation notes
│   ├── TECHNICAL_GUIDE.md             # Full mathematical derivations
│   ├── lin-chang-csvt-01.pdf          # Original paper (IEEE TCSVT 2001)
│   └── My_sythesized_data_stuff/      # Synthetic evaluation dataset pipeline
│       ├── generate_dataset.py        # Generates compressed / tampered / clean images
│       ├── verification.py            # Runs authentication over the dataset
│       ├── DATASET_GENERATION.md      # Dataset design documentation
│       ├── input_images/              # Raw input images
│       └── generated/                 # Processed output images
│
├── LISTA/                             # Sparse recovery and algorithm unrolling
│   ├── ista.py                        # ISTA (Iterative Shrinkage-Thresholding)
│   ├── fista.py                       # FISTA (Fast ISTA with Nesterov momentum)
│   ├── lista.py                       # LISTA (Learned ISTA via unrolling)
│   ├── unrolled_fista.py              # Unrolled FISTA (L-FISTA)
│   ├── generate_synthetic_data.py     # Sparse signal dataset generator
│   └── papers/                        # Reference papers
│
├── SimCLR/                            # Contrastive learning for image forensics
│   ├── SimCLR.py                      # ResNet-50 + projection head architecture
│   ├── NT_Xent.py                     # NT-Xent contrastive loss
│   ├── augmentation.py                # Augmentation pipeline (forensics-tuned)
│   ├── train.py                       # Training loop (LARS, cosine LR schedule)
│   ├── evaluation.py                  # Linear probing on frozen backbone
│   ├── simclr_metrics.py              # Embedding similarity metrics
│   ├── single_pair_eval.py            # Per-pair evaluation
│   └── plot_vectors.py                # Embedding visualizer
│
├── ECC Image Authentication Papers/   # Reference papers on ECC-based semi-fragile auth
├── Contrastive_Learning/Papers/       # Reference papers on contrastive / self-supervised learning
├── XAI/Papers/                        # Reference papers on explainability (GradCAM, SHAP)
├── Compression_Basics.docx            # Notes on compression fundamentals
└── deepsum.docx                       # Summaries of key ECC authentication papers
```

---

## Module 1: Lin & Chang Implementation

### What It Does

A complete Python implementation of the semi-fragile authentication scheme from:

> Ching-Yung Lin and Shih-Fu Chang, *"A Robust Image Authentication Method Distinguishing JPEG Compression from Malicious Manipulation,"* IEEE Transactions on Circuits and Systems for Video Technology, Vol. 11, No. 2, 2001.

### The Mathematical Invariant

The scheme is built on **Theorem 1** from the paper. For two DCT coefficient vectors `y_a` and `y_b` from non-overlapping 8×8 blocks of an image, and a JPEG quantization table `q`:

- If `y_a[k] > y_b[k]` before quantization → `Q_a[k] ≥ Q_b[k]` after quantization
- If `y_a[k] < y_b[k]` before quantization → `Q_a[k] ≤ Q_b[k]` after quantization
- If `y_a[k] = y_b[k]` before quantization → `Q_a[k] = Q_b[k]` after quantization

This holds for any quantization table and any number of recompression iterations — because all blocks are divided by the *same* quantization table, so relative orderings are preserved. Malicious replacement of pixel content changes DCT coefficient values in a way that breaks this relationship with high probability.

**Theorem 2** extends this from just the sign to magnitude: using multiple threshold levels (binary subdivision of the dynamic range), differences can be bounded with increasing precision, providing a stronger, multi-bit feature per block pair.

### How the System Works

**Signing phase:**
1. Convert image to grayscale. Divide into non-overlapping 8×8 blocks. Compute 2D DCT for each block in zigzag order.
2. Pair up blocks (even-odd mapping, seeded for secrecy). The seed is a secret parameter that serves as part of the security mechanism.
3. Run three nested loops — over threshold sets `k = 0..3`, over all block pairs, and over 10 selected low/mid-frequency DCT positions — computing a binary feature bit at each position via Theorem 2. This produces a binary feature string.
4. Record the mean DC value per coefficient position across all blocks, to defeat constant-intensity-shift attacks that would otherwise leave all differences unchanged.
5. Serialize the features as JSON and sign with HMAC-SHA256 (the paper specifies RSA public-key encryption; HMAC is used here for simplicity and can be swapped).

**Verification phase:**
1. Load the received image. Recompute DCT blocks with the same parameters.
2. Re-run the same three nested loops to generate a new feature string from the received image.
3. Compare bit-by-bit against the stored feature string (Proposition 1 from the paper).
4. If the mismatch rate is below a tolerance threshold (~5%), the image is authenticated. Mismatched block pairs are flagged and mapped back to spatial image regions for localization.

The tolerance threshold `δ` is tunable by the user: set it to 0 for images expected to be compressed only once, to 2–5 for images that may be recompressed multiple times. This is the mechanism that makes the scheme adaptive to the full range of acceptable manipulations — filtering noise, rounding errors, and scaling artifacts all manifest as small, bounded deviations within the tolerance, while malicious content replacement causes large, widespread mismatches.

### What Passes and What Fails

| Manipulation | Result | Why |
|---|---|---|
| JPEG compression (any ratio) | ✅ Passes | DCT coefficient relationships preserved by Theorem 1 |
| Multiple recompressions | ✅ Passes | Theorem 1 holds across iterations |
| Integer rounding noise | ✅ Passes | Bounded deviation, within tolerance δ |
| Image filtering / enhancement | ✅ Passes | Primarily affects high-freq coefficients; selected positions are low/mid-freq |
| Constant brightness change | ✅ Passes | DC mean comparison catches only large shifts; small ones fall within tolerance |
| Scaling + resize back | ✅ Passes | Resampling noise is Gaussian and bounded; within tolerance |
| Content replacement / splice | ❌ Detected | DCT structure of replaced region breaks stored feature codes |
| Cloning (in-image copy-paste) | ❌ Detected | Block pair relationships break at tampered region |
| Pixel-level tampering (large area) | ❌ Detected | High mismatch rate across affected block pairs |

### Usage

```python
from image_auth import ImageAuthenticator

auth = ImageAuthenticator(
    block_size=8,
    num_thresholds=4,
    num_coefficients=10,
    base_power=7        # T = 2^7 = 128, threshold levels: 0, 128, 64, 32
)

# Sign the original image
features = auth.extract_features('original.jpg', block_pair_seed=42)
signature_hex, signature_data = auth.generate_signature(features)

# Authenticate a received image (may be JPEG-compressed, filtered, scaled, etc.)
result = auth.authenticate('received.jpg', signature_data, tolerance_bound=2.0)

if result['authenticated']:
    print("✅ Authentic — content is intact (may have been compressed or filtered)")
else:
    print(f"❌ Manipulated — {result['tampering_confidence']:.1%} confidence")
    print(f"   Suspicious block pairs: {result['manipulated_pairs']}")
```

```bash
# Self-contained demo: signs a random image, tests original, JPEG-compressed, and tampered versions
python image_auth.py
```

### Synthetic Evaluation Dataset

`My_sythesized_data_stuff/` contains a controlled pipeline that tests all three cases the authentication system needs to handle correctly. Images are processed by index `N`:

- **Prime N** → JPEG compressed at quality 80 — should **pass** authentication
- **Multiple of 5** → Passed through unchanged — should **pass** authentication
- **All others** → Random patch from another image copy-pasted in — should **fail** authentication

```bash
cd "Lin&Chang implementation/My_sythesized_data_stuff"
python generate_dataset.py   # Populates generated/ with all three image types
python verification.py       # Runs authentication on all and reports results
```

### Evaluation on CASIA

`verification_casia.py` and `verification_casia_gt.py` run the authenticator against the CASIA tampered image database, comparing authentication decisions against ground-truth tamper masks. Results are logged to `casia_full_report.csv` and `casia_diagnosis_report.csv`.

---

## Module 2: LISTA (Learned ISTA / Algorithm Unrolling)

### Why It's Here

Sparse signal recovery is deeply connected to the image authentication problem. The DCT is itself a sparsifying transform — natural images have most of their energy concentrated in a small number of low-frequency coefficients. Understanding what information is preserved under linear transformations (like DCT + quantization) requires understanding sparse representations.

This module implements the progression from classical iterative sparse recovery to learned, unrolled versions — a general framework for turning iterative signal processing algorithms into trainable neural networks.

### What's Implemented

**ISTA** (`ista.py`) — Classic Iterative Shrinkage-Thresholding Algorithm. Solves:
```
min_x  ||Ax - y||²  +  λ||x||₁
```
Configured for recovery of K=5-sparse signals from M=30 measurements of an N=100-dimensional signal. Uses a deterministic θ schedule (logarithmic decay from 0.5 to 0.001) to ensure convergence — adaptive schedules were found to be unstable for large batch sizes.

**FISTA** (`fista.py`) — Fast ISTA with Nesterov momentum. Converges in O(1/k²) vs O(1/k) for plain ISTA.

**LISTA** (`lista.py`) — Algorithm Unrolling of ISTA. Each iteration of ISTA becomes a trainable layer with its own weight matrices and threshold parameter. Trained supervised (measurements Y → sparse signal X_true) on 50k synthetic examples. Initialized from the ISTA update rule, then fine-tuned end-to-end.

**Unrolled FISTA** (`unrolled_fista.py`) — Same unrolling idea applied to FISTA with momentum (L-FISTA).

### Configuration

| Parameter | Value |
|---|---|
| Signal dimension N | 100 |
| Measurements M | 30 |
| Sparsity K | 5 |
| Unrolled layers | 10 |
| Training samples | 50,000 |
| Validation samples | 2,000 |
| Optimizer | Adam, lr=1e-3 |
| Epochs | 250 |

### Running

```bash
cd LISTA
python generate_synthetic_data.py   # Creates dataset.pt
python ista.py                      # Classical ISTA baseline
python fista.py                     # FISTA baseline
python lista.py                     # Train and evaluate LISTA
python unrolled_fista.py            # Train and evaluate L-FISTA
```

All scripts output convergence plots (NMSE vs. iterations or epochs).

---

## Module 3: SimCLR for Image Forensics

### What It Does

Applies SimCLR (Chen et al., 2020) to learn image representations for forensic purposes. The motivation: a model trained contrastively to produce similar embeddings for different augmentations of the *same* authentic image should place deepfakes and tampered images out-of-distribution relative to authentic content. This is a learned, data-driven alternative to hand-crafted feature codes.

### Architecture

```
Input Image
    │
    ├── Augmentation View 1 ──► ResNet-50 Encoder f(·) ──► h₁ (2048-d) ──► Projection g(·) ──► z₁ (128-d)
    │                                   ↕ shared weights
    └── Augmentation View 2 ──► ResNet-50 Encoder f(·) ──► h₂ (2048-d) ──► Projection g(·) ──► z₂ (128-d)
                                                                         NT-Xent Loss on (z₁, z₂)
```

The projection head is a 2-layer MLP: `2048 → 2048 (ReLU) → 128`. After pretraining, the head is discarded — only the 2048-d representation `h` is used downstream.

### Augmentation Pipeline

The pipeline is specifically tuned for image forensics and goes beyond the standard SimCLR augmentations to include operations that a tampered or AI-generated image might realistically undergo:

| Augmentation | Probability | Forensics Rationale |
|---|---|---|
| Random Resized Crop | Always | Core SimCLR augmentation |
| Random Horizontal Flip | 50% | Standard |
| Color Jitter | 80% | Simulates brightness/contrast edits — an acceptable manipulation |
| Random Grayscale | 20% | Standard |
| Gaussian Blur | 50% | Simulates post-processing |
| **Random JPEG Compression** (quality 10–100) | 50% | Explicitly encodes robustness to the key acceptable manipulation |
| **Gaussian Noise** | 50% | Simulates sensor noise vs. synthetic artifacts |
| Random Erasing (Cutout) | 50% | Simulates partial occlusion |

Each image is passed through this pipeline twice independently to generate a positive pair. The key forensic insight is that if the model learns to produce similar embeddings for all these augmented versions of authentic images, forensically anomalous images (deepfakes, spliced content) should fall outside this learned cluster.

### Training

```bash
cd SimCLR
# Expects a dataset directory in ./Faces/ organized as ImageFolder
python train.py
```

Paper-accurate hyperparameters (adapted for single GPU):

| Parameter | Value |
|---|---|
| Batch size | 128 |
| Epochs | 2000 |
| Temperature τ | 0.5 |
| Optimizer | LARS (from `torch_optimizer`) |
| Base LR | 0.3 × (batch_size / 256) |
| LR schedule | Linear warmup + cosine decay |
| Weight decay | 1e-6 (excluded from bias and BN layers) |

Model was trained for 1030 epochs with checkpoints saved as `simclr_checkpoint_epoch_*.pth`.

### Evaluation

**Embedding similarity check** — measures cosine similarity between authentic images and their augmented views:
```bash
python simclr_metrics.py
# Expects test/real/ and test/augmentations/ with matched image pairs
```

**Linear probing** — freezes the ResNet-50 backbone, trains only a linear `nn.Linear(2048, 2)` head on labeled authentic vs. tampered data:
```bash
python evaluation.py
```

**Attention heatmaps** — GradCAM-style visualizations of which image regions the encoder focuses on. Pre-computed heatmaps are included for CASIA tampered images (`176_CASIA_heatmap.png`), face datasets (`actresses_heatmap.png`, `heatmap_utk_200.png`).

---

## Paper Library

### ECC Image Authentication (`ECC Image Authentication Papers/`)

Papers on semi-fragile authentication using error-correcting codes and watermarking. `deepsum.docx` contains detailed reading notes on several of these. Key works:

- **Lin & Chang (2001)** — the core paper implemented here
- **Joint SVD and QR Codes for Image Authentication** — SVD-based block fingerprinting with embedded QR code watermarks for tamper localization and block recovery without needing the original image
- **AACI: Approximate Authentication and Correction of Images** — hybrid hard/soft authentication: DC coefficients get strict MAC + Reed-Solomon protection, AC coefficients get soft hash-based tolerance; the system can not only detect but also correct and localize tampering
- **A Quantitative Semi-Fragile JPEG2000 Image Authentication System** — introduces the concept of a "Lowest Authenticable Bit Rate" (LABR) for explicit, quantitative control over which levels of recompression are considered acceptable vs. unacceptable
- Semi-fragile watermarking schemes, SVD, ECC+PKI frameworks, VQ-compressed image authentication, contourlet-domain feature extraction

### Contrastive Learning (`Contrastive_Learning/Papers/`)

SimCLR, MoCo, BYOL, DINO, MAE, CLIP, CPC, NCE, SimSiam.

### LISTA (`LISTA/papers/`)

Original LISTA (Gregor & LeCun, 2010), ISTA foundations, Ada-LISTA, FISTA-Net, NA-ALISTA, Compressed Sensing reference.

### XAI (`XAI/Papers/`)

GradCAM (Selvaraju et al., 2017), SHAP (Lundberg & Lee, 2017).

---

## Dependencies

### Lin & Chang
```bash
pip install numpy scipy pillow
```

### LISTA
```bash
pip install torch numpy matplotlib
```

### SimCLR
```bash
pip install torch torchvision torch_optimizer tqdm pillow numpy
```

---

## Setup

```bash
git clone https://github.com/AlmostHeroicGuy/Image-Authentication.git
cd Image-Authentication

# Lin & Chang demo (no dataset needed, generates its own test image)
cd "Lin&Chang implementation"
pip install numpy scipy pillow
python image_auth.py

# LISTA experiments
cd ../LISTA
pip install torch numpy matplotlib
python generate_synthetic_data.py
python lista.py

# SimCLR (requires face images in SimCLR/Faces/ as ImageFolder)
cd ../SimCLR
pip install torch torchvision torch_optimizer tqdm
python train.py
```

---

## Notes

- The Lin & Chang implementation uses HMAC-SHA256 in place of the paper's specified RSA public-key encryption. Replacing it with RSA is straightforward using `pycryptodome` or `cryptography`.
- SimCLR was trained on an RTX 5090. On smaller GPUs, reduce `batch_size` in `train.py` and scale the learning rate proportionally (`lr = 0.3 * batch_size / 256`).
- The LISTA scripts use a deterministic logarithmic θ schedule instead of adaptive scheduling — this was necessary to ensure convergence at large batch sizes.
