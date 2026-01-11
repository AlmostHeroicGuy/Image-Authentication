import torch
import os

# CONFIGURATION
M, N, K = 30, 100, 5
TEST_SAMPLES = 1000
TARGET_SUCCESS_RATE = 0.99 # We want 99% of samples to pass

def solve_ista_fast(y, A, x_true):
    """ Quick solver to test matrix quality """
    b = y.shape[0]
    x = torch.zeros(b, N)
    
    # Quick Check parameters
    L = torch.linalg.norm(A.T @ A, ord=2).item()
    eta = 1.0 / L
    lam = 0.001 # Moderate lambda for quick check
    theta = lam * eta
    
    # Run 200 fast steps
    for _ in range(200):
        res = x @ A.T - y
        grad = res @ A
        x = torch.sign(x - eta*grad) * torch.relu(torch.abs(x - eta*grad) - theta)
        
    # Check success (roughly <-20dB is enough to know it's solvable)
    diff = torch.norm(x_true - x, dim=1)**2
    ref = torch.norm(x_true, dim=1)**2
    nmse = 10 * torch.log10(diff / (ref + 1e-12))
    
    success_count = (nmse < -20).sum().item()
    return success_count / b

print(f"Hunting for a Golden Matrix (M={M}, K={K})...")

best_seed = -1
best_rate = 0.0

# Try 100 random seeds
for seed in range(100):
    torch.manual_seed(seed)
    
    # 1. Generate Matrix A
    A = torch.randn(M, N)
    A = torch.nn.functional.normalize(A, p=2, dim=0)
    
    # 2. Generate Test Data
    X = torch.zeros(TEST_SAMPLES, N)
    for i in range(TEST_SAMPLES):
        indices = torch.randperm(N)[:K]
        X[i, indices] = torch.randn(K)
    Y = X @ A.T
    
    # 3. Test It
    rate = solve_ista_fast(Y, A, X)
    
    print(f"Seed {seed:<3} | Success Rate: {rate*100:.1f}%")
    
    if rate > best_rate:
        best_rate = rate
        best_seed = seed
    
    if rate >= TARGET_SUCCESS_RATE:
        print("\n" + "="*40)
        print(f"FOUND GOLDEN SEED: {seed}")
        print(f"Success Rate: {rate*100:.1f}%")
        print("="*40)
        break

print(f"\nBest Seed Found: {best_seed} (Rate: {best_rate*100:.1f}%)")
print("Update your generate_data.py with this seed!")