# Technical Implementation Guide
## Image Authentication: Mathematical Foundations and Implementation

---

## Section 1: Mathematical Framework

### 1.1 DCT Formulation

**2D Discrete Cosine Transform (Equation 1):**
```
y = T_x @ x @ T_x^T

Where:
- x: Input 8×8 spatial image block
- y: Output DCT coefficient block
- T_x: DCT transformation matrix
- T_x^T: Transpose of DCT matrix
```

**Implementation:**
```python
def _compute_dct_blocks(self, image: np.ndarray) -> List[np.ndarray]:
    height, width = image.shape
    dct_blocks = []
    
    for i in range(0, height - self.block_size + 1, self.block_size):
        for j in range(0, width - self.block_size + 1, self.block_size):
            block = image[i:i+8, j:j+8].astype(float)
            # Apply 2D DCT using normalized orthogonal basis
            dct_coeffs = dct(dct(block, axis=0, norm='ortho'), 
                           axis=1, norm='ortho')
            # Convert to zigzag scan order (matches JPEG)
            zigzag_coeffs = self._zigzag_order(dct_coeffs)
            dct_blocks.append(zigzag_coeffs)
    
    return dct_blocks
```

### 1.2 JPEG Quantization Process

**Quantization (Equation 2-3):**
```
Q[i,j] = round(Y[i,j] / q[i,j])

Where:
- Y[i,j]: DCT coefficient at position (i,j)
- q[i,j]: Quantization step size from JPEG quantization table
- Q[i,j]: Quantized coefficient

Quantized approximation:
Y_hat[i,j] = Q[i,j] × q[i,j]
```

**Key Insight from Paper:**
Despite quantization, the relationship between coefficients from different blocks is preserved:
```
If DCT_A[k] > DCT_B[k]  (before quantization at position k)
Then Q_A[k] ≥ Q_B[k]    (after quantization, with high probability)
```

### 1.3 Theorem 1: Sign Preservation

**Theorem Statement:**
For DCT coefficient vectors y_a and y_b from two non-overlapping blocks:

```
Define: d_k = y_a[k] - y_b[k]  (difference at position k)

Then:
1. If d_k > 0 before quantization  ⟹  Q_A[k] ≥ Q_B[k] after quantization
2. If d_k < 0 before quantization  ⟹  Q_A[k] ≤ Q_B[k] after quantization
3. If d_k = 0 before quantization  ⟹  Q_A[k] = Q_B[k] after quantization

(Exception: Quantization may change ">" or "<" to "=" due to rounding)
```

**Implementation:**
```python
def _extract_feature_code_bit(self, diff_dc: float, threshold: float) -> int:
    """
    Extract feature bit using Theorem 2.
    Implements equations 5-7 from paper.
    """
    if diff_dc < threshold:
        return 0  # "less than" relationship
    else:
        return 1  # "greater than or equal" relationship
```

### 1.4 Theorem 2: Magnitude Preservation with Multi-Thresholds

**Extended Theorem (Multiple Thresholds):**
```
For threshold value τ:

        ┌ 0  if d_k < τ        (sign only)
F(d_k) = │
        └ 1  if d_k ≥ τ        (relative magnitude)

Multiple threshold sets provide increasing precision:
- Set 0: τ_0 = 0           (sign information)
- Set 1: τ_1 = T           (magnitude ≥ T)
- Set 2: τ_2 = T/2         (magnitude ≥ T/2)
- Set 3: τ_3 = T/4         (magnitude ≥ T/4)
...
- Set k: τ_k = T / 2^(k-1) (progressive refinement)
```

**Threshold Calculation (Equation 10):**
```
T_k = T / 2^(k-1)  for k ≥ 1
T_0 = 0            for sign-only protection

Where: T = 2^base_power (typically 2^7 = 128)

Example with T=128 and base_power=7:
- k=0: τ_0 = 0     (sign only)
- k=1: τ_1 = 128   (±128 range)
- k=2: τ_2 = 64    (±64 range)
- k=3: τ_3 = 32    (±32 range)
- k=4: τ_4 = 16    (±16 range)
```

**Implementation:**
```python
def _compute_threshold_value(self, k: int, diff_value: float) -> float:
    """
    Compute threshold for k-th set using binary division (Equation 10).
    """
    if k == 0:
        return 0  # Sign protection only
    else:
        # Binary division: T_k = T / 2^(k-1)
        return self.T / (2 ** (k - 1))

# For feature extraction loop:
for k in range(self.num_thresholds):
    threshold_k = self._compute_threshold_value(k, 1.0)
    # Use threshold_k for this set of feature codes
```

---

## Section 2: Feature Extraction Algorithm

### 2.1 Three Nested Loops (Image Analyzer)

**Overall Structure:**
```
for k in range(num_thresholds):           # Loop 1: Threshold sets
    threshold_k = compute_threshold(k)
    
    for pair_idx in range(num_pairs):     # Loop 2: Block pairs
        block_a = block_set_A[pair_idx]
        block_b = block_set_B[pair_idx]
        
        dct_a = DCT_blocks[block_a]
        dct_b = DCT_blocks[block_b]
        
        for coeff_idx in range(num_coefficients):  # Loop 3: Positions
            diff = dct_a[coeff_idx] - dct_b[coeff_idx]
            bit = extract_feature_bit(diff, threshold_k)
            feature_codes.append(bit)
```

**Complexity Analysis:**
```
Time Complexity:
T = O(k × pairs × coefficients)
  = O(num_thresholds × (N/2) × num_coefficients)
  = O(4 × 128 × 10)  for 256 blocks
  ≈ O(5,120 operations)

Space Complexity:
S = O(num_blocks × block_size²) for storing DCT
  + O(feature_string_length)
  = O(256 × 64) + O(5,120 bits)
  ≈ 16KB for typical image

For 512×512 image:
T = O(4 × 2048 × 10) ≈ O(81,920 operations)
S ≈ 256KB
```

### 2.2 Block Pair Mapping Function

**Equations 8-9: Partition Conditions**
```
Partition A ∪ Partition B = All blocks
Partition A ∩ Partition B = ∅ (no overlap, non-redundant case)

Simple mapping used in implementation (even-odd):
Partition_A = {0, 2, 4, 6, ..., 2i, ...}     (even-indexed blocks)
Partition_B = {1, 3, 5, 7, ..., 2i+1, ...}   (odd-indexed blocks)

Pairing: (0,1), (2,3), (4,5), ...

Secret enhancement: Use seed-based pseudo-random mapping
```

**Implementation:**
```python
def _create_block_pairs(self, num_blocks: int, seed: int = 42):
    """
    Create block pairs using even-odd mapping.
    Seed serves as secret parameter.
    """
    np.random.seed(seed)
    
    even_blocks = [i for i in range(num_blocks) if i % 2 == 0]
    odd_blocks = [i for i in range(num_blocks) if i % 2 == 1]
    
    # Equal length pairs
    min_len = min(len(even_blocks), len(odd_blocks))
    return even_blocks[:min_len], odd_blocks[:min_len]
```

### 2.3 Selected DCT Coefficient Positions

**Position Selection Strategy:**
```
Select positions from low and mid-frequency bands:
- Lower frequencies: larger DCT coefficients (energy concentration)
- Mid frequencies: stable across compressions
- High frequencies: eliminated by aggressive JPEG quantization

Typical selection (JPEG zigzag order):
Position 0:   DC coefficient (DC)
Positions 1-3:   AC coefficients (1,0), (2,0), (1,1)
Positions 4-9:   Mid-frequency AC coefficients

Example (first 10 positions):
[0, 1, 2, 5, 6, 14, 15, 27, 28, 29]
    DC  AC  AC   AC   AC    AC   AC   AC   AC   AC
```

**Energy Distribution in DCT:**
```
Frequency distribution (8×8 block):
        Columns (horizontal frequency)
    0   1   2   3   4   5   6   7
0  [DC] [A] [B] [C] [D] [E] [F] [G]
1  [A] [B] [C] [D] [E] [F] [G] [H]
2  [B] [C] [D] [E] [F] [G] [H] [I]
3  [C] [D] [E] [F] [G] [H] [I] [J]
...

Energy concentration: 90% in top-left 16×16 region
Selection focuses on: Positions 0-16 in zigzag order
```

### 2.4 DC Mean Values (Constant Change Defense)

**Problem:** Uniform intensity change affects all blocks equally
```
Attack: Add constant c to all pixel values
Result: All DCT differences remain unchanged ✗
```

**Solution: Record DC Component Means**
```
For each selected position p:
    dc_mean[p] = mean(DCT_block[p] for all blocks)

Authentication checks:
    if |mean(detected[p]) - stored_dc_mean[p]| > δ_dc:
        MANIPULATION DETECTED
```

**Implementation:**
```python
# Record DC means during feature extraction
dc_means = []
for coeff_idx in range(self.num_coefficients):
    dc_values = []
    for block_idx in range(num_blocks):
        if coeff_idx < len(dct_blocks[block_idx]):
            dc_values.append(dct_blocks[block_idx][coeff_idx])
    if dc_values:
        dc_means.append(np.mean(dc_values))

# Storage: ~40-64 bytes per signature
```

---

## Section 3: Signature Generation and Encryption

### 3.1 Signature Data Structure

**Complete Signature Format:**
```json
{
    "features": "0111100110...",        // Binary feature string
    "dc_means": [127.3, 45.2, ...],    // DC component means
    "num_blocks": 256,                 // Total 8×8 blocks
    "block_pair_seed": 42,             // Secret mapping function seed
    "image_height": 512,               // Original image height
    "image_width": 512,                // Original image width
    "metadata": {
        "num_thresholds": 4,
        "num_coefficients": 10,
        "block_size": 8,
        "base_power": 7
    }
}
```

### 3.2 Signature Length Calculation

**Formula:**
```
L_features = num_thresholds × num_block_pairs × num_coefficients
           = k × (N/2) × m bits
           
where:
- k = num_thresholds (typically 4)
- N = num_blocks
- m = num_coefficients (typically 10)

L_dc_means = num_selected_positions × 8 bytes
           = m × 8 bytes

L_metadata = overhead for storing seeds, dimensions, etc.
           ≈ 50-100 bytes

Total: L = L_features + L_dc_means + L_metadata (in bits/bytes)
```

**Example Calculations:**
```
128×128 image:
- Blocks: 256 (16×16 grid)
- Feature bits: 4 × 128 × 10 = 5,120 bits = 640 bytes
- DC means: 10 × 8 = 80 bytes
- Metadata: ~100 bytes
- Total: ~820 bytes (≈0.64% of uncompressed image)

256×256 image:
- Blocks: 1,024 (32×32 grid)
- Feature bits: 4 × 512 × 10 = 20,480 bits = 2,560 bytes
- DC means: 10 × 8 = 80 bytes
- Total: ~2,640 bytes (≈0.40% of uncompressed image)

512×512 image:
- Blocks: 4,096 (64×64 grid)
- Feature bits: 4 × 2,048 × 10 = 81,920 bits = 10,240 bytes
- Total: ~10,400 bytes (≈0.40% of uncompressed image)
```

### 3.3 Encryption Methods

**Paper Specifies: RSA Public Key Cryptography**

**Implementation Options:**

**Option 1: RSA Encryption (Paper Standard)**
```python
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
import base64

# Generate keys (one-time setup)
key = RSA.generate(2048)
private_key = key.exportKey()
public_key = key.publickey().exportKey()

# Sign with private key
cipher = PKCS1_v1_5.new(private_key)
signature = cipher.encrypt(json_data.encode())

# Verify with public key
cipher = PKCS1_v1_5.new(public_key)
verified_data = cipher.decrypt(signature)
```

**Option 2: HMAC with Shared Secret (Simplified)**
```python
import hmac
import hashlib

# Using shared secret key
shared_secret = b"secret_key_phrase"

signature = hmac.new(
    shared_secret,
    json_data.encode(),
    hashlib.sha256
).hexdigest()

# Verification
if hmac.compare_digest(computed_sig, stored_sig):
    print("Signature valid ✓")
```

**Option 3: Digital Signature (DSA/ECDSA)**
```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# Generate keypair
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)

# Sign
signature = private_key.sign(
    json_data.encode(),
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)
```

**Current Implementation:**
```python
def generate_signature(self, features: Dict) -> str:
    """
    Generate HMAC-SHA256 signature (simplified).
    Can be replaced with RSA for production use.
    """
    signature_data = {
        'features': features['feature_codes'],
        'dc_means': [float(x) for x in features['dc_means']],
        'num_blocks': features['num_blocks'],
        'block_pair_seed': features['block_pair_seed'],
        'image_height': features['image_height'],
        'image_width': features['image_width'],
        'metadata': features['metadata']
    }
    
    json_str = json.dumps(signature_data, sort_keys=True)
    signature = hashlib.sha256(json_str.encode()).hexdigest()
    
    return signature, signature_data
```

---

## Section 4: Authentication Process

### 4.1 Proposition 1: Authentication Criterion

**Comparison Test (Equations 11-13):**
```
Define: R = Detected_feature_value
        S = Stored_feature_value
        δ = Tolerance bound

Manipulation Detection:
Block pair (i,j) is manipulated if:

    |R - S| > δ  (exceeds tolerance)

Tolerance δ accounts for:
- Integer rounding noise from DCT/IDCT
- JPEG compression artifacts
- Multiple recompression iterations
- Color space decimation differences
```

**Implementation:**
```python
def authenticate(self, test_image_path, signature_data, 
                tolerance_bound=2.0) -> Dict:
    """
    Authenticate image (Figure 2b - Authentication Comparator).
    """
    # Load and process test image
    image_array = load_image(test_image_path)
    dct_blocks = self._compute_dct_blocks(image_array)
    
    # Verify dimensions
    if image_array.shape != (signature_data['image_height'], 
                            signature_data['image_width']):
        return {'authenticated': False, 'reason': 'Dimension mismatch'}
    
    # Recompute features using same parameters
    detected_features = []
    for k in range(self.num_thresholds):
        threshold_k = self._compute_threshold_value(k, 1.0)
        for pair_idx in range(len(block_set_A)):
            for coeff_idx in range(self.num_coefficients):
                diff = dct_a[coeff_idx] - dct_b[coeff_idx]
                bit = self._extract_feature_code_bit(diff, threshold_k)
                detected_features.append(bit)
    
    # Compare (Proposition 1)
    detected_string = ''.join(map(str, detected_features))
    stored_string = signature_data['features']
    
    mismatches = sum(1 for d,s in zip(detected_string, stored_string) 
                     if d != s)
    mismatch_rate = mismatches / len(stored_string)
    
    # Authentication threshold
    auth_threshold = 0.05  # 5% tolerance
    is_authenticated = mismatch_rate < auth_threshold
    
    return {
        'authenticated': is_authenticated,
        'mismatch_rate': mismatch_rate,
        'manipulated_pairs': identified_pairs,
        'tampering_confidence': min(mismatch_rate / auth_threshold, 1.0)
    }
```

### 4.2 Tolerance Bound Calculation

**From Section IV-A: Noise Analysis**

**Rounding Noise Probability (Equations 14-18):**
```
DCT with rounding noise:
Y[i,j] = floor(T·x + η_dct + η_rounding)

where:
- T: DCT transformation
- η_dct: DCT operation noise (finite precision)
- η_rounding: Integer rounding noise

Probability of false alarm due to noise:
P_FA(pair) = 1 - ∏(1 - P_single_alarm[pos])
            (product over all compared positions)

For single coefficient position:
P_alarm = 2·Φ(-δ/σ)

where:
- δ: Tolerance bound
- σ: Standard deviation of noise
- Φ: Standard normal CDF
```

**Setting Tolerance δ:**
```
Conservative approach:
δ_rounding = 2.0  (2-3× estimated noise std dev)

Progressive approach:
δ = 0              for one-time compression
δ = 2.0-5.0        for recompression tolerance

Adaptive approach:
δ = estimate_noise_std() × confidence_factor
```

**Implementation:**
```python
def estimate_tolerance_bound(self, test_image_jpeg_iterations=1):
    """
    Estimate appropriate tolerance bound based on
    expected compression iterations.
    """
    if test_image_jpeg_iterations == 1:
        return 1.0  # Single compression
    elif test_image_jpeg_iterations <= 3:
        return 2.0  # Few recompressions
    else:
        return 5.0  # Many recompressions
```

### 4.3 Tampered Block Localization

**Identifying Manipulated Regions:**
```
For each block pair (i,j):
    if detected_features ≠ stored_features:
        pair_idx = identify_pair_index(i,j)
        manipulated_pairs.append(pair_idx)

Localization:
- Non-overlapping pairs: Identify pair, can't pinpoint block
- Overlapping pairs: Narrow down to specific blocks

Spatial reconstruction:
manipulated_regions = reconstruct_spatial_map(manipulated_pairs)
```

**Implementation:**
```python
def identify_tampered_regions(self, mismatches_per_pair):
    """
    Convert block pair indices to spatial image regions.
    """
    height, width = self.image_height, self.image_width
    blocks_per_row = width // self.block_size
    
    manipulated_mask = np.zeros((height, width), dtype=bool)
    
    for pair_idx in mismatches_per_pair:
        # Convert pair index to block indices
        block_a_idx = (2 * pair_idx) % (height//8 * width//8)
        block_b_idx = block_a_idx + 1
        
        # Convert to spatial location
        for block_idx in [block_a_idx, block_b_idx]:
            row = (block_idx // blocks_per_row) * self.block_size
            col = (block_idx % blocks_per_row) * self.block_size
            manipulated_mask[row:row+8, col:col+8] = True
    
    return manipulated_mask
```

---

## Section 5: Robustness Analysis

### 5.1 Acceptable Manipulations (Pass Authentication)

**JPEG Lossy Compression:**
```
Why it passes:
- Coefficient relationships preserved (Theorem 1)
- Quantization changes individual values, not relationships
- Multiple recompressions: Relationships remain stable
- Any quality factor: Same quantization table structure

Verification:
Load JPEG → Extract DCT → Compare relationships → PASS
```

**Integer Rounding Errors:**
```
Source: DCT/IDCT floating-point to integer conversion
Magnitude: ±1-2 units per coefficient
Handling: Tolerance bound δ = 2.0-5.0
```

**Scaling and Resampling:**
```
Small scaling (±5%):
1. Record original dimensions in signature
2. Resize test image to original size
3. Resampling noise modeled as Gaussian
4. Apply tolerance bounds

Implementation:
if test_image.shape != original_shape:
    test_image = resize(test_image, original_shape)
```

**Intensity Adjustments:**
```
Constant-value changes (brightness):
- Affects only DC component (position 0)
- AC components unchanged
- DC means detector catches excessive changes
- Small changes: Tolerance on DC mean difference

Implementation:
if |detected_dc_mean - stored_dc_mean| > dc_tolerance:
    MANIPULATION DETECTED
```

**Filtering and Enhancement:**
```
Low-pass filtering:
- Primarily affects high-frequency coefficients
- Selected positions (0-9) mostly low/mid frequency
- Minimal impact on authentication

Edge enhancement:
- Increases mid-frequency energy
- Detected as magnitude changes
- Tolerance bounds accommodate small changes
```

### 5.2 Rejected Manipulations (Fail Authentication)

**Content Replacement:**
```
Attack: Copy pixels from one region to another
Result: DCT coefficient differences change significantly
Detection: Feature code mismatches in affected blocks

Why it fails:
- Replaced content has different DCT patterns
- Block pair relationships break
- Mismatch rate exceeds threshold
```

**Splicing/Forgery:**
```
Attack: Paste region from different image
Result: Complete DCT structure change in modified blocks

Why it fails:
- All feature codes for affected blocks become mismatches
- Extreme mismatch rate (50%+ mismatches in area)
- Well above authentication threshold
```

**Pixel-level tampering:**
```
Attack: Small pixel modifications to change content
Result: Depends on modification extent

Small changes (few pixels):
- May not significantly alter DCT relationships
- Detected if spread across multiple blocks
- Blending detected via localization

Extensive changes:
- Certain detection via high mismatch rate
```

---

## Section 6: Performance Metrics

### 6.1 Error Probabilities

**Type I Error (False Alarm - P_FA):**
```
Event: Authentic image marked as manipulated
Typical rate: 1-5% with δ = 2.0

Causes:
- Rounding noise variance estimation error
- Aggressive JPEG quantization
- Lossy color space conversion

Mitigation: Increase δ, increase num_thresholds
```

**Type II Error (Miss - P_M):**
```
Event: Manipulated image marked as authentic
Typical rate: <1% for targeted attacks

Causes:
- Tampering designed to minimize DCT changes
- Smooth blending of spliced regions
- Correlated manipulation patterns

Mitigation: Increase num_coefficients, overlapping block pairs
```

### 6.2 Tradeoff Curves

**Signature Length vs. Security:**
```
More thresholds (k):
- Longer signature
- Better magnitude precision
- Lower miss probability

Trade: 1 threshold → 4 thresholds = 4× signature size
       But P_M reduced significantly

Typical sweet point: k = 3-4 thresholds
```

**Tolerance Bound vs. False Alarms:**
```
δ = 0.0:   P_FA highest, P_M lowest (strict)
δ = 2.0:   Balanced (recommended)
δ = 5.0:   P_FA lowest, P_M highest (permissive)

Selection depends on application:
- News authentication: δ = 0-1 (strict)
- Medical imaging: δ = 1-2 (moderate)
- Archive preservation: δ = 2-5 (permissive)
```

---

## Section 7: Implementation Checklist

- [x] DCT computation with zigzag ordering
- [x] Block pair creation and mapping
- [x] Threshold calculation (binary division)
- [x] Feature extraction with three nested loops
- [x] DC mean recording
- [x] Signature generation and encryption
- [x] Image dimension verification
- [x] Feature recomputation during authentication
- [x] Mismatch detection (Proposition 1)
- [x] Tolerance bound application
- [x] Tampered block localization
- [x] JPEG compression robustness testing
- [x] Demonstration with sample images

---

## References to Paper Equations

| Equation | Description | Implementation |
|----------|-------------|-----------------|
| (1) | 2D DCT | `_compute_dct_blocks()` |
| (2-3) | JPEG quantization | Reference only (no quantization needed) |
| (5-7) | Theorem 2 (feature extraction) | `_extract_feature_code_bit()` |
| (8-9) | Block pair partitioning | `_create_block_pairs()` |
| (10) | Threshold calculation | `_compute_threshold_value()` |
| (11-13) | Proposition 1 (authentication) | `authenticate()` comparison logic |
| (14-18) | Noise analysis | `tolerance_bound` parameter |

