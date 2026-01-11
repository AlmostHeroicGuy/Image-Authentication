import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
FILENAME = "dataset.pt"  # As per your instruction
TEST_BATCH_SIZE = 100     # Only 10 samples
MAX_ITER = 2000
LAMBDA = 0.0012855
TOL = 1e-10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running FISTA on: {device}")

# ==========================================
# 1. LOAD DATA
# ==========================================
if not os.path.exists(FILENAME):
    raise FileNotFoundError("dataset.pt not found. Run generation script first.")

data = torch.load(FILENAME)
A = data["A"].to(device)
# Slice first 10 samples for testing
X_true = data["X"][:TEST_BATCH_SIZE].to(device)
Y = data["Y"][:TEST_BATCH_SIZE].to(device)

# ==========================================
# 2. FISTA ALGORITHM
# ==========================================
def fista(y, A, lam, max_iter, tol=1e-6):
    b, n = y.shape[0], A.shape[1]
    
    # Lipschitz Setup
    L = torch.linalg.norm(torch.matmul(A.T, A), ord=2).item()
    eta = 1.0 / L
    theta = lam * eta
    
    # Initialize variables
    x = torch.zeros(b, n, device=device)
    z = torch.zeros(b, n, device=device) # Momentum point
    t = 1.0                              # Momentum scalar
    
    loss_curve = []
    
    for _ in range(max_iter):
        # 1. Gradient Step on Z (Predictor)
        res = z @ A.T - y
        grad = res @ A
        
        # 2. Proximal Step
        x_next = z - eta * grad
        x_next = torch.sign(x_next) * torch.relu(torch.abs(x_next) - theta)
        
        # 3. Momentum Update
        t_next = (1 + np.sqrt(1 + 4*t**2)) / 2
        momentum = (t - 1) / t_next
        z = x_next + momentum * (x_next - x)
        
        # Track Loss (MSE)
        current_loss = torch.norm(z - x, p=2) # Convergence proxy
        loss_curve.append(current_loss.item())
        
        # Update pointers
        t = t_next
        x = x_next.clone()

        # Stopping criterion: residual norm ||y - A x||_F
        residual = y - torch.matmul(x, A.T)
        residual_norm = torch.norm(residual).item()
        if residual_norm < tol:
            break
        
    return x, loss_curve

# ==========================================
# 3. EXECUTION & PLOT
# ==========================================
print("Running FISTA...")
x_est, loss = fista(Y, A, LAMBDA, MAX_ITER, tol=TOL)

# Metrics
error_norm = torch.norm(X_true - x_est, p=2, dim=1) ** 2
true_norm = torch.norm(X_true, p=2, dim=1) ** 2
nmse = 10 * torch.log10(error_norm / (true_norm + 1e-10))
avg_nmse = nmse.mean().item()

print(f"FISTA Avg NMSE: {avg_nmse:.2f} dB")

# Visualization
idx = torch.argmin(nmse).item() # Best sample
idx2 = torch.argmax(nmse).item() # Worst sample
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss, label='Convergence proxy')
plt.title('FISTA Convergence')
plt.yscale('log')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.stem(X_true[idx].cpu().numpy(), linefmt='k-', markerfmt='ko', basefmt='k-', label='True')
plt.stem(x_est[idx].cpu().numpy(), linefmt='r--', markerfmt='rx', basefmt='k-', label='FISTA')
plt.title(f'Best reconstruction (NMSE: {nmse[idx]:.2f} dB)')
plt.legend()
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.stem(X_true[idx2].cpu().numpy(), linefmt='k-', markerfmt='ko', basefmt='k-', label='True')
plt.stem(x_est[idx2].cpu().numpy(), linefmt='r--', markerfmt='rx', basefmt='k-', label='FISTA')
plt.title(f'Worst reconstruction (NMSE: {nmse[idx2]:.2f} dB)')
plt.legend()
plt.tight_layout()
plt.show()