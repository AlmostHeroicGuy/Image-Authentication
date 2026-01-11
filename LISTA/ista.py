import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

FILENAME = "dataset.pt"
TEST_BATCH_SIZE = 1000 
L_CONST = 7.2561  #LIPSCHITZ CONSTANT
ETA = 0.137816    # 1 / L_CONST

# ==========================================
# 1. LOAD CLEAN DATA
# ==========================================
if not os.path.exists(FILENAME):
    raise FileNotFoundError("dataset.pt not found.")
    
data = torch.load(FILENAME, map_location=device, weights_only=False)
A = data["A"].to(device)
X_true = data["X"][:TEST_BATCH_SIZE].to(device)
Y_clean = torch.matmul(X_true, A.T)

# ==========================================
# 2. METRIC HELPERS
# ==========================================
def compute_support_metrics(x_est, x_true, threshold=1e-4):
    """
    Calculates Hit Rate (Recall) and Precision of support recovery.
    """
    # 1. Identify Non-Zero Indices (Support)
    pred_support = torch.abs(x_est) > threshold
    true_support = torch.abs(x_true) > threshold
    
    # 2. Calculate Intersection (Correctly identified spikes)
    intersection = (pred_support & true_support).sum(dim=1).float()
    
    # 3. Denominators
    true_count = true_support.sum(dim=1).float()  # Total actual spikes (K=5)
    pred_count = pred_support.sum(dim=1).float()  # Total predicted spikes
    
    # 4. Metrics
    # Hit Rate: What % of true spikes did we find?
    hit_rate = (intersection / (true_count + 1e-9)).mean().item()
    
    # Precision: What % of our predictions were actually correct? (Avoids hallucinations)
    precision = (intersection / (pred_count + 1e-9)).mean().item()
    
    return hit_rate, precision

# ==========================================
# 3. INSTRUMENTED ISTA SOLVER
# ==========================================
def solve_final_ista_tracked(y, A, x_true):
    b, n = y.shape[0], A.shape[1]
    
    # Schedule
    lambdas = np.logspace(np.log10(0.5), np.log10(1e-9), 12)
    
    x = torch.zeros(b, n, device=device)
    
    # History logs
    history = {
        'steps': [],
        'nmse': [],
        'hit_rate': [],
        'precision': []
    }
    
    total_steps = 0
    print(f"Running ISTA on {b} samples...")
    
    for i, lam in enumerate(lambdas):
        eta = ETA
        theta = lam * eta
        
        # Iterations per stage
        # First 11 stages: 200 steps. Last stage: 2000 steps.
        steps = 200 if i < len(lambdas)-1 else 2000
        
        for _ in range(steps):
            # Standard ISTA Step
            res = x @ A.T - y
            grad = res @ A
            z = x - eta * grad
            x = torch.sign(z) * torch.relu(torch.abs(z) - theta)
            
            total_steps += 1
            
            # --- TRACKING (Every 50 steps to save time) ---
            if total_steps % 50 == 0:
                # 1. NMSE
                diff = torch.norm(x_true - x, dim=1)**2
                ref = torch.norm(x_true, dim=1)**2
                batch_nmse = 10 * torch.log10(diff / (ref + 1e-12)).mean().item()
                
                # 2. Support Metrics
                hr, prec = compute_support_metrics(x, x_true)
                
                history['steps'].append(total_steps)
                history['nmse'].append(batch_nmse)
                history['hit_rate'].append(hr * 100)   # Percentage
                history['precision'].append(prec * 100) # Percentage

    return x, history

# ==========================================
# 4. EXECUTION & PLOTTING
# ==========================================
x_est, hist = solve_final_ista_tracked(Y_clean, A, X_true)

# --- A. Final Stats (Histogram Data) ---
diff = torch.norm(X_true - x_est, dim=1)**2
ref = torch.norm(X_true, dim=1)**2
nmse_list = 10 * torch.log10(diff / (ref + 1e-12))
mean_nmse = nmse_list.mean().item()
median_nmse = nmse_list.median().item()
success_rate = (nmse_list < -40).sum().item() / TEST_BATCH_SIZE * 100

print("\n" + "="*40)
print(f"FINAL RESULTS")
print("="*40)
print(f"Mean NMSE:    {mean_nmse:.2f} dB")
print(f"Median NMSE:  {median_nmse:.2f} dB")
print(f"Success Rate: {success_rate:.1f}%")
print("-" * 40)

# --- B. PLOTS ---
plt.figure(figsize=(15, 10))

# Plot 1: Histogram (Distribution)
plt.subplot(2, 2, 1)
plt.hist(nmse_list.cpu().numpy(), bins=50, color='blue', alpha=0.7)
plt.axvline(x=mean_nmse, color='red', linestyle='--', label=f'Mean: {mean_nmse:.1f} dB')
plt.axvline(x=median_nmse, color='green', linestyle='-', label=f'Median: {median_nmse:.1f} dB')
plt.title("Final NMSE Distribution")
plt.xlabel("NMSE (dB)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Avg NMSE vs Steps
plt.subplot(2, 2, 2)
plt.plot(hist['steps'], hist['nmse'], 'b-', linewidth=2)
plt.title("Avg NMSE vs Steps")
plt.xlabel("Cumulative Steps")
plt.ylabel("NMSE (dB)")
plt.axhline(y=-50, color='r', linestyle='--', label='Target (-50dB)')
plt.legend()
plt.grid(True)

# Plot 3: Hit Rate (Recall) vs Steps
plt.subplot(2, 2, 3)
plt.plot(hist['steps'], hist['hit_rate'], 'g-', linewidth=2, label='Hit Rate (Recall)')
plt.title("Hit Rate (% True Spikes Found)")
plt.xlabel("Cumulative Steps")
plt.ylabel("Percentage (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(True)

# Plot 4: Precision vs Steps (Effectiveness check)
plt.subplot(2, 2, 4)
plt.plot(hist['steps'], hist['precision'], 'purple', linewidth=2, label='Precision')
plt.title("Precision (% Predictions that are Correct)")
plt.xlabel("Cumulative Steps")
plt.ylabel("Percentage (%)")
plt.ylim(0, 105)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""
PARAMETER EXPLANATIONS:
-----------------------
1. L_CONST (7.2561): 
   The Lipschitz constant of the gradient, equal to the maximum eigenvalue of (A^T * A). 
   It represents the maximum curvature of the loss landscape. If we step further than 2/L, 
   the algorithm will diverge (explode).

2. ETA (0.137816): 
   The Step Size (Learning Rate), set strictly to 1/L. This is the optimal safe speed 
   limit for standard ISTA. It guarantees monotonic convergence (the error goes down every 
   single step) without oscillation.

3. LAMBDA SCHEDULE (np.logspace(0.5, 1e-9, 12)):
   - Start (0.5): High regularization. Finds the "rough location" of the 5 spikes 
     while suppressing all noise. Acts as a "coarse focus."
   - End (1e-9): Effectively zero. This removes the "Shrinkage Bias" inherent to L1. 
     By relaxing lambda to zero, we allow the algorithm to match the exact height of 
     the true signal, pushing NMSE from -30 dB (biased) to -100 dB (unbiased).
   - Logarithmic Spacing: Ensures the algorithm spends equal effort solving every 
     order of magnitude, preventing it from getting stuck in local minima early on.

4. STEP SCHEDULE (200 vs 2000):
   - Early Stages (200 steps): We only need a "warm start" (rough guess) to pass 
     to the next stage. Converging fully here is a waste of compute.
   - Final Stage (2000 steps): Once lambda is zero, the problem becomes "flat." 
     We need massive iteration count here to grind the tiny numerical errors 
     down to machine precision (-50 dB target).
"""