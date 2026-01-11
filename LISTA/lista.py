import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
FILENAME = "dataset.pt"
LAYERS = 10
EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-3
TRAIN_SPLIT = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running LISTA on: {device}")

# ==========================================
# 1. LOAD & SPLIT DATA
# ==========================================
if not os.path.exists(FILENAME):
        raise FileNotFoundError("dataset.pt not found.")

# Load dataset onto CPU first (safe across devices) then move tensors to `device`
data = torch.load(FILENAME, map_location='cpu')
# We treat X as the "Target" (Labels) and Y as "Input" (Features)
full_X = data["X"].to(device) # (10000, 100)
full_Y = data["Y"].to(device) # (10000, 30)

total_samples = full_X.shape[0]
train_size = int(total_samples * TRAIN_SPLIT)

# Split
train_X, test_X = full_X[:train_size], full_X[train_size:]
train_Y, test_Y = full_Y[:train_size], full_Y[train_size:]

# Create DataLoaders
train_dataset = torch.utils.data.TensorDataset(train_Y, train_X)
test_dataset = torch.utils.data.TensorDataset(test_Y, test_X)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Data Loaded: {train_size} Training Samples, {total_samples - train_size} Test Samples")

# ==========================================
# 2. FIXED LISTA MODEL DEFINITION
# ==========================================
class LISTALayer(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        # We ONLY learn the threshold here.
        # W1 and W2 are shared (Tied) and passed in during forward().
        self.theta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, y, W1, W2):
        # FIX 1: Removed 'self.' because W1 and W2 are passed as arguments
        # W1 and W2 are nn.Linear layers passed from the parent Net
        z = W1(y) + W2(x) 
        return torch.sign(z) * torch.relu(torch.abs(z) - self.theta)

class LISTANet(nn.Module):
    def __init__(self, m, n, layers):
        super().__init__()
        self.n = n
        self.m = m
        
        # 1. Define LAYERS (Containers for thresholds)
        self.layers = nn.ModuleList([LISTALayer(m, n) for _ in range(layers)])

        # 2. Define TIED WEIGHTS (Shared across all layers)
        self.W1 = nn.Linear(m, n, bias=False) 
        self.W2 = nn.Linear(n, n, bias=False) 

        # 3. GENERIC INITIALIZATION (Critical for Learning from Scratch)
        # This is NOT using Matrix A. This is using standard RNN best practices.
        with torch.no_grad():
            # Initialize W2 as Identity + Noise.
            # This ensures x_k+1 starts as a copy of x_k (gradient highway),
            # rather than a random scrambling of x_k.
            self.W2.weight.data.copy_(torch.eye(n))
            self.W2.weight.data.add_(torch.randn(n, n) * 0.001)
            
            # Initialize W1 with Xavier (Standard for inputs)
            nn.init.xavier_uniform_(self.W1.weight)
            
    def forward(self, y):
        batch_size = y.shape[0]
        x = torch.zeros(batch_size, self.n, device=y.device)
        
        # Pass the SAME shared W1 and W2 to every layer
        for layer in self.layers:
            x = layer(x, y, self.W1, self.W2)
        return x
# ==========================================
# 3. TRAINING LOOP 
# ==========================================
model = LISTANet(m=30, n=100, layers=LAYERS).to(device)

# 1. Use a Scheduler (Critical for learning from scratch)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
criterion = nn.MSELoss()

print(f"\nStarting Training (Random Init - Learning Physics from Scratch)...")
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_y, batch_x in train_loader:
        optimizer.zero_grad()
        x_pred = model(batch_y)
        loss = criterion(x_pred, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    # Update Learning Rate based on loss
    scheduler.step(avg_loss)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

# ==========================================
# 4. TESTING & METRICS
# ==========================================
print("\nEvaluating on Test Set...")
model.eval()
all_nmse = []

with torch.no_grad():
    for batch_y, batch_x in test_loader:
        x_est = model(batch_y)
        
        # Calculate NMSE (dB)
        error_norm = torch.norm(batch_x - x_est, p=2, dim=1) ** 2
        true_norm = torch.norm(batch_x, p=2, dim=1) ** 2
        nmse = 10 * torch.log10(error_norm / (true_norm + 1e-10))
        all_nmse.append(nmse)

final_nmse = torch.cat(all_nmse).mean().item()
print(f"FINAL TEST NMSE: {final_nmse:.2f} dB")

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))

# Plot 1: Training Loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, 'b-', label='MSE Loss')
plt.title(f'LISTA Training ({LAYERS} Layers)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot 2: Reconstruction (First sample of last batch)
plt.subplot(1, 2, 2)
# Grab one sample from the last test batch
x_t = batch_x[0].cpu().numpy()
x_e = x_est[0].cpu().numpy()

plt.stem(x_t, linefmt='k-', markerfmt='ko', basefmt='k-', label='True')
plt.stem(x_e, linefmt='r--', markerfmt='rx', basefmt='k-', label='LISTA')
plt.title(f'Test Sample Reconstruction')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()