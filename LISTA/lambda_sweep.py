import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from ista import ista, get_metrics

# Configuration
FILENAME = "dataset.pt"
TEST_BATCH_SIZE = 10
MAX_ITER = 25000
TOL = 1e-10

if not os.path.exists(FILENAME):
    raise FileNotFoundError("dataset.pt not found. Run generate_synthetic_data.py first.")

data = torch.load(FILENAME, map_location='cpu')
A = data['A'].to('cpu')
X_all = data['X'].to('cpu')
Y_all = data['Y'].to('cpu')

# Use first TEST_BATCH_SIZE samples for the sweep
X = X_all[:TEST_BATCH_SIZE].to(A.device)
Y = Y_all[:TEST_BATCH_SIZE].to(A.device)

# Lambda sweep
lams = np.linspace(0.001285, 0.0012875, 10) # 0.0012855
avg_nmses = []

print(f"Running lambda sweep over {len(lams)} values...")
for i, lam in enumerate(lams):
    x_est, history = ista(Y, A, float(lam), MAX_ITER, X, tol=TOL)
    final_nmses, _ = get_metrics(X, x_est)
    avg_nmse = torch.nanmean(final_nmses).item()
    avg_nmses.append(avg_nmse)
    if (i % 20) == 0:
        print(f"{i+1}/{len(lams)} lambda={lam:.4f} avg NMSE={avg_nmse:.2f} dB")

# Plot lambda vs avg NMSE
plt.figure(figsize=(8,5))
plt.plot(lams, avg_nmses, '-o', markersize=3)
plt.xlabel('Lambda')
plt.ylabel('Avg NMSE (dB)')
plt.title(f'Lambda sweep (batch {TEST_BATCH_SIZE}, tol={TOL})')
plt.grid(True)
plt.tight_layout()
plt.show()

print('Done.')
