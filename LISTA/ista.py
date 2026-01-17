import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

N_SIGNAL = 100
M_MEASUREMENTS = 30
K_SPARSITY = 5
BATCH_SIZE = 100
MAX_ITERATIONS = 200  # < 200 Iterations Goal
TOLERANCE_G = 0.1

# ==========================================
# 1. GENERATE DATA (Same as before)
# ==========================================
torch.manual_seed(42)
np.random.seed(42)

A = torch.randn(M_MEASUREMENTS, N_SIGNAL, device=device)
A = A / torch.norm(A, dim=0, keepdim=True) # Normalize columns

X_true = torch.zeros(BATCH_SIZE, N_SIGNAL, device=device)
for i in range(BATCH_SIZE):
    indices = torch.randperm(N_SIGNAL, device=device)[:K_SPARSITY]
    X_true[i, indices] = torch.rand(K_SPARSITY, device=device) * 8.0 + 2.0 # [2, 10]

Y_clean = torch.matmul(X_true, A.T)

# ==========================================
# 2. SETUP SOLVER
# ==========================================
# L Constant
L_const = torch.linalg.norm(A.T @ A, ord=2).item()

# Aggressive Step Size
# The theoretical limit is 2/L according to the original ISTA paper. 
ETA = 2.4968/ L_const # Passing the limit is giving good results in practice for small batch. 2.5 was giving inf in NMSE   
# 2.4968 for 100 batch
# The theoretically unsafe values of eta not working for a large batch size

# --- THE FIX: DETERMINISTIC SCHEDULE ---
# Instead of adaptive, we FORCE theta to go from 0.5 down to 0.001.
# This breaks the "Feedback Loop" and guarantees convergence.
theta_schedule = np.logspace(np.log10(0.5), np.log10(0.001), MAX_ITERATIONS)

# ==========================================
# 3. METRIC HELPER
# ==========================================
def compute_metrics(x_est, x_true, g=0.1):
    threshold = 1e-4
    pred_supp = torch.abs(x_est) > threshold
    true_supp = torch.abs(x_true) > threshold
    
    hits = (pred_supp & true_supp).sum(dim=1).float()
    hit_rate = (hits / (true_supp.sum(dim=1) + 1e-9)).mean().item() * 100
    precision = (hits / (pred_supp.sum(dim=1) + 1e-9)).mean().item() * 100
    
    # Amp Accuracy
    diff_ratio = (x_true - x_est).abs() / (x_true.abs() + 1e-9)
    # Check if error is <= 10% AND it is a support index
    accurate_spikes = ((diff_ratio <= g) & true_supp).sum().item()
    total_spikes = true_supp.sum().item()
    amp_acc = (accurate_spikes / total_spikes) * 100 if total_spikes > 0 else 0
    
    return hit_rate, precision, amp_acc

# ==========================================
# 4. ITERATIVE SOLVER (OPTIMIZED WITH FULL TRACKING)
# ==========================================
x = torch.zeros(BATCH_SIZE, N_SIGNAL, device=device)

# Initialize expanded history for the "beautiful graphs"
history = {
    'steps': [],
    'nmse': [],
    'theta': [],
    'hit_rate': [],
    'precision': [],
    'amp_acc': []
}

print(f"{'Iter':<6} {'Theta':<10} {'NMSE (dB)':<10} {'AmpAcc %':<10}")
print("-" * 40)

for k in range(MAX_ITERATIONS):
    # 1. Get Theta from Schedule
    theta = theta_schedule[k]
    
    # 2. ISTA Step
    res = x @ A.T - Y_clean
    grad = res @ A
    z = x - ETA * grad
    x = torch.sign(z) * torch.relu(torch.abs(z) - theta)
    
    # 3. Track Metrics (Every step for smooth graphs)
    diff = torch.norm(X_true - x, dim=1)**2
    ref = torch.norm(X_true, dim=1)**2
    nmse = 10 * torch.log10(diff / (ref + 1e-12)).mean().item()
    
    # Compute detailed metrics
    hr, prec, amp_acc = compute_metrics(x, X_true, g=TOLERANCE_G)
    
    # Store in history
    history['steps'].append(k)
    history['nmse'].append(nmse)
    history['theta'].append(theta)
    history['hit_rate'].append(hr)
    history['precision'].append(prec)
    history['amp_acc'].append(amp_acc)
    
    # Print progress every 20 steps
    if (k+1) % 20 == 0:
        print(f"{k+1:<6} {theta:<10.4f} {nmse:<10.2f} {amp_acc:<10.1f}")

# ==========================================
# 5. RESULTS & VISUALIZATION (Updated)
# ==========================================
# 1. Compute Final Per-Sample Metrics
final_diff = torch.norm(X_true - x, dim=1)**2
final_ref = torch.norm(X_true, dim=1)**2
nmse_per_sample = 10 * torch.log10(final_diff / (final_ref + 1e-12))
nmse_values = nmse_per_sample.detach().cpu().numpy()

# Statistics
mean_nmse = np.mean(nmse_values)
median_nmse = np.median(nmse_values)
std_nmse = np.std(nmse_values)

final_hr = history['hit_rate'][-1]
final_prec = history['precision'][-1]
final_amp = history['amp_acc'][-1]

print("\n" + "="*40)
print(f"FINAL RESULTS ({MAX_ITERATIONS} Iterations)")
print("="*40)
print(f"Mean NMSE:          {mean_nmse:.2f} dB")
print(f"Median NMSE:        {median_nmse:.2f} dB")
print(f"Std Dev NMSE:       {std_nmse:.2f} dB")
print(f"Support Recall:     {final_hr:.1f}%")
print(f"Support Precision:  {final_prec:.1f}%")
print(f"Amplitude Accuracy: {final_amp:.1f}%")
print("="*40)

# --- THE PLOT (Theta removed, Signal Plot added back) ---
plt.figure(figsize=(15, 8))

# 1. NMSE vs Steps
plt.subplot(3, 2, 1)
plt.plot(history['steps'], history['nmse'], 'b-', linewidth=2)
plt.title("Average NMSE Convergence", fontsize=12, fontweight='bold')
plt.xlabel("Iteration")
plt.ylabel("NMSE (dB)")
plt.grid(True, alpha=0.3)

# 2. Support Recall
plt.subplot(3, 2, 2)
plt.plot(history['steps'], history['hit_rate'], 'g-', linewidth=2)
plt.title("Support Recall (Location)", fontsize=12, fontweight='bold')
plt.xlabel("Iteration")
plt.ylabel("Recall (%)")
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.axhline(y=100, color='k', linestyle=':', alpha=0.5)

# 3. Support Precision
plt.subplot(3, 2, 3)
plt.plot(history['steps'], history['precision'], 'purple', linewidth=2)
plt.title("Support Precision", fontsize=12, fontweight='bold')
plt.xlabel("Iteration")
plt.ylabel("Precision (%)")
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.axhline(y=100, color='k', linestyle=':', alpha=0.5)

# 4. Amplitude Accuracy
plt.subplot(3, 2, 4)
plt.plot(history['steps'], history['amp_acc'], 'orange', linewidth=2)
plt.title(f"Amplitude Accuracy (Error < {TOLERANCE_G*100:.0f}%)",
          fontsize=12, fontweight='bold')
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.axhline(y=100, color='k', linestyle=':', alpha=0.5)

# 5. NMSE Distribution (Histogram)
plt.subplot(3, 2, 5)
plt.hist(nmse_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(mean_nmse, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_nmse:.1f}')
plt.axvline(median_nmse, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_nmse:.1f}')
plt.title(f"NMSE Distribution (Std Dev: {std_nmse:.1f} dB)", fontsize=12, fontweight='bold')
plt.xlabel("NMSE (dB)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 6. Signal Reconstruction (Sample #0)
plt.subplot(3, 2, 6)
worst_idx = np.argmax(nmse_values)
best_idx = np.argmin(nmse_values)
print(f"Worst Sample Index: {worst_idx} with NMSE {nmse_values[worst_idx]:.2f} dB")
print(f"Best Sample Index: {best_idx} with NMSE {nmse_values[best_idx]:.2f} dB")
sample_idx = 99
# 90 is -18dB and 0 is -23.5dB, 99 and 98 are very good
x_true_np = X_true[sample_idx].detach().cpu().numpy()
x_est_np = x[sample_idx].detach().cpu().numpy()
plt.stem(np.arange(N_SIGNAL), x_true_np, linefmt='g-', markerfmt='go', label='True')
plt.stem(np.arange(N_SIGNAL), x_est_np, linefmt='b--', markerfmt='bx', label='Est')
plt.title(f"Sample #{sample_idx} Reconstruction ({nmse_values[sample_idx]:.1f} dB)", fontsize=12, fontweight='bold')
plt.xlabel("Index")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ista_stats.png', dpi=150)
print("Figure saved as 'ista_stats.png'")
plt.show()