"""
LISTA (Learned ISTA) - PyTorch Implementation (FIXED)
=====================================================
1. Architecture: nn.ModuleList of custom LISTALayers.
2. Initialization: Smart ISTA weights (Start strong, then learn).
3. Training: Supervised (Y -> X_true).
"""

import torch
import torch.nn as nn
import torch.optim as optim
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

# Training Config
# Reduced slightly for speed, increase if you have a fast GPU
TRAIN_SIZE = 50000
VAL_SIZE = 2000
BATCH_SIZE = 250
EPOCHS = 250
LEARNING_RATE = 1e-3

# LISTA Config
NUM_LAYERS = 10
TOLERANCE_G = 0.1  

# ==========================================
# 1. GENERATE DATA
# ==========================================
torch.manual_seed(42)
np.random.seed(42)

# Generate System Matrix A
A = torch.randn(M_MEASUREMENTS, N_SIGNAL, device=device)
A = A / torch.norm(A, dim=0, keepdim=True)

# Helper to generate X and Y
def generate_batch(batch_size):
    X = torch.zeros(batch_size, N_SIGNAL, device=device)
    for i in range(batch_size):
        indices = torch.randperm(N_SIGNAL, device=device)[:K_SPARSITY]
        X[i, indices] = torch.rand(K_SPARSITY, device=device) * 0.5 + 0.8
    Y = X @ A.T
    return X, Y

# Generate Datasets
X_train, Y_train = generate_batch(TRAIN_SIZE)
X_val, Y_val = generate_batch(VAL_SIZE)

print(f"Training Set: {X_train.shape}")
print(f"Validation Set: {X_val.shape}")

# ==========================================
# 2. LISTA MODEL DEFINITION (FIXED)
# ==========================================
class LISTALayer(nn.Module):
    """
    A single unrolled iteration of ISTA.
    """
    def __init__(self, n, m, A, L, lambda_val):
        super(LISTALayer, self).__init__()
        
        # --- Smart Initialization (ISTA Weights) ---
        # W1 = (1/L) * A^T
        # W2 = I - (1/L) * A^T * A
        
        # We detach() to ensure these are leaf tensors that start tracking gradients new
        W1_init = (A.T / L).detach()
        W2_init = (torch.eye(n, device=device) - (A.T @ A) / L).detach()
        theta_init = (torch.tensor(lambda_val / L, device=device)).detach()

        self.W1 = nn.Parameter(W1_init)
        self.W2 = nn.Parameter(W2_init)
        self.theta = nn.Parameter(theta_init)

    def forward(self, x, y):
        # Linear Step: z = x @ W2.T + y @ W1.T
        z = x @ self.W2.T + y @ self.W1.T
        
        # Soft Thresholding
        theta = torch.abs(self.theta) # Constraint: Threshold must be positive
        return torch.sign(z) * torch.relu(torch.abs(z) - theta)

class LISTA(nn.Module):
    def __init__(self, A, num_layers=16, lambda_val=0.01):
        super(LISTA, self).__init__()
        m, n = A.shape
        
        # Compute Lipschitz constant L = largest eigenvalue of A^T A
        # (Using operator norm approx)
        L = torch.linalg.norm(A.T @ A, ord=2).item()
        
        # Create the Unrolled Layers
        self.layers = nn.ModuleList([
            LISTALayer(n, m, A, L, lambda_val) for _ in range(num_layers)
        ])

    def forward(self, y):
        batch_size = y.shape[0]
        # Initialize x with 0
        x = torch.zeros(batch_size, N_SIGNAL, device=device)
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x, y)
            
        return x
    
# Initialize Model
model = LISTA(A, NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# ==========================================
# 3. TRAINING LOOP
# ==========================================
history = {
    'steps': [],
    'nmse': [],
    'hit_rate': [],
    'precision': [],
    'amp_acc': [],
    'lr': []
}

# Scheduler: Decay LR every 5 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

print(f"{'Epoch':<6} {'Loss':<10} {'Val NMSE':<10} {'AmpAcc %':<10} {'LR':<10}")
print("-" * 55)

for epoch in range(EPOCHS):
    
    # --- Train Step ---
    model.train()
    # Batch Training
    # (Simple full-batch for demo, but typically you'd loop over batches here)
    num_batches = (TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE
    epoch_loss = 0
    
    permutation = torch.randperm(TRAIN_SIZE)
    
    for i in range(0, TRAIN_SIZE, BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train[indices], Y_train[indices]
        
        optimizer.zero_grad()
        x_pred = model(batch_y)
        loss = criterion(x_pred, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # --- Scheduler Step ---
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    # --- Validation Step ---
    model.eval()
    with torch.no_grad():
        x_val_est = model(Y_val)
        
        # NMSE
        diff = torch.norm(X_val - x_val_est, dim=1)**2
        ref = torch.norm(X_val, dim=1)**2
        val_nmse = 10 * torch.log10(diff / (ref + 1e-12)).mean().item()
        
        # Metrics
        threshold = 1e-4
        pred_supp = torch.abs(x_val_est) > threshold
        true_supp = torch.abs(X_val) > threshold
        
        hits = (pred_supp & true_supp).sum(dim=1).float()
        hr = (hits / (true_supp.sum(dim=1) + 1e-9)).mean().item() * 100
        prec = (hits / (pred_supp.sum(dim=1) + 1e-9)).mean().item() * 100
        
        # Amp Acc
        diff_ratio = (X_val - x_val_est).abs() / (X_val.abs() + 1e-9)
        accurate_spikes = ((diff_ratio <= TOLERANCE_G) & true_supp).sum().item()
        total_spikes = true_supp.sum().item()
        amp_acc = (accurate_spikes / total_spikes) * 100 if total_spikes > 0 else 0
        
        history['steps'].append(epoch)
        history['nmse'].append(val_nmse)
        history['hit_rate'].append(hr)
        history['precision'].append(prec)
        history['amp_acc'].append(amp_acc)
        history['lr'].append(current_lr)
        
        print(f"{epoch+1:<6} {loss.item():<10.6f} {val_nmse:<10.2f} {amp_acc:<10.1f} {current_lr:<10.2e}")

# ==========================================
# 4. RESULTS & VISUALIZATION (UNCOMMENTED)
# ==========================================
model.eval()
with torch.no_grad():
    x_val_est = model(Y_val)
    nmse_values = (10 * torch.log10(torch.norm(X_val - x_val_est, dim=1)**2 / (torch.norm(X_val, dim=1)**2 + 1e-12))).cpu().numpy()

mean_nmse = np.mean(nmse_values)
median_nmse = np.median(nmse_values)
std_nmse = np.std(nmse_values)

print("\n" + "="*40)
print(f"FINAL RESULTS")
print("="*40)
print(f"Mean NMSE:          {mean_nmse:.2f} dB")
print(f"Median NMSE:        {median_nmse:.2f} dB")
print(f"Support Recall:     {history['hit_rate'][-1]:.1f}%")
print("="*40)

# --- PLOTTING ---
plt.figure(figsize=(12, 9))

# 1. NMSE vs Epochs
plt.subplot(3, 2, 1)
plt.plot(history['steps'], history['nmse'], 'b-', linewidth=2)
plt.title("Val NMSE vs Epochs", fontsize=10, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("NMSE (dB)")
plt.grid(True, alpha=0.3)

# 2. Support Recall
plt.subplot(3, 2, 2)
plt.plot(history['steps'], history['hit_rate'], 'g-', linewidth=2)
plt.title("Support Recall", fontsize=10, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Recall (%)")
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)

# 3. Support Precision
plt.subplot(3, 2, 3)
plt.plot(history['steps'], history['precision'], 'purple', linewidth=2)
plt.title("Support Precision", fontsize=10, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Precision (%)")
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)

# 4. Amp Accuracy
plt.subplot(3, 2, 4)
plt.plot(history['steps'], history['amp_acc'], 'orange', linewidth=2)
plt.title(f"Amplitude Accuracy (<{TOLERANCE_G*100:.0f}%)", fontsize=10, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)

# 5. NMSE Distribution
plt.subplot(3, 2, 5)
plt.hist(nmse_values, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
plt.axvline(mean_nmse, color='red', linestyle='--', label=f'Mean:{mean_nmse:.0f}')
plt.title(f"NMSE Hist (Std:{std_nmse:.1f})", fontsize=10, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Signal Reconstruction
plt.subplot(3, 2, 6)
sample_idx = 0 
x_true_np = X_val[sample_idx].detach().cpu().numpy()
x_est_np = x_val_est[sample_idx].detach().cpu().numpy()
plt.stem(np.arange(N_SIGNAL), x_true_np[:N_SIGNAL], linefmt='g-', markerfmt='go', label='True')
plt.stem(np.arange(N_SIGNAL), x_est_np[:N_SIGNAL], linefmt='b--', markerfmt='bx', label='LISTA')
plt.title(f"Sample #{sample_idx} Recon", fontsize=10, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig('lista_stats.png', dpi=150)
print("Figure saved as 'lista_stats.png'")
plt.show()