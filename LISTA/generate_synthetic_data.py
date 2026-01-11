import torch
import os

# ==========================================
# CONFIGURATION
# ==========================================
M = 30           # Measurements
N = 100          # Features
K = 5            # Sparsity
SAMPLES = 10000  # Total dataset size
NOISE_LEVEL = 0.0 # Added Noise (Sigma)
FILENAME = "dataset.pt"

def generate_data():
    print(f"Generating dataset (M={M}, N={N}, K={K}, Sigma={NOISE_LEVEL})...")
    
    # --- 1. GENERATE MATRIX A (FIXED SEED 69) ---
    print("Setting Seed 69 for Matrix A...")
    torch.manual_seed(69)
    
    # Generate A
    A = torch.randn(M, N)
    # Normalize columns (L2 norm = 1)
    A = torch.nn.functional.normalize(A, p=2, dim=0)
    
    # --- 2. GENERATE X and Y (RANDOM) ---
    print("Re-randomizing seed for Vectors X...")
    torch.seed() # Random seed for X
    
    # Generate Sparse X
    X = torch.zeros(SAMPLES, N)
    for i in range(SAMPLES):
        indices = torch.randperm(N)[:K]
        values = torch.randn(K)
        X[i, indices] = values
        
    # Generate Measurements Y = X @ A.T
    Y = torch.matmul(X, A.T)
    '''
    # --- 3. ADD NOISE ---
    print(f"Adding Gaussian Noise (std={NOISE_LEVEL})...")
    noise = torch.randn_like(Y) * NOISE_LEVEL
    Y = Y + noise
    '''
    # --- 4. SAVE ---
    data = {
        "A": A,
        "X": X,
        "Y": Y,
        "config": {"M": M, "N": N, "K": K, "NOISE": NOISE_LEVEL}
    }
    
    torch.save(data, FILENAME)
    print(f"SUCCESS: Saved {SAMPLES} samples to '{FILENAME}'")

if __name__ == "__main__":
    generate_data()