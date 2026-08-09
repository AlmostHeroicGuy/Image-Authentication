import re
import matplotlib.pyplot as plt
import numpy as np

# Change this to your .out file path
filename = "resnet10-135150.out"

batch_losses = []
batch_steps = []

epoch_losses = []
epochs = []

with open(filename, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Match batch losses
        batch_match = re.search(
            r"Epoch\s+\[(\d+)/\d+\]\s+Batch\s+\[\s*(\d+)/\d+\]\s+Loss:\s+([-+]?\d*\.?\d+)",
            line
        )

        if batch_match:
            epoch = int(batch_match.group(1))
            batch = int(batch_match.group(2))
            loss = float(batch_match.group(3))

            batch_steps.append(len(batch_steps))
            batch_losses.append(loss)

        # Match epoch average losses
        epoch_match = re.search(
            r"Epoch\s+(\d+)/\d+\s+complete\s+\|\s+Avg Loss:\s+([-+]?\d*\.?\d+)",
            line
        )

        if epoch_match:
            epoch = int(epoch_match.group(1))
            avg_loss = float(epoch_match.group(2))

            epochs.append(epoch)
            epoch_losses.append(avg_loss)


print(f"Found {len(batch_losses)} batch losses")
print(f"Found {len(epoch_losses)} epoch losses")

# ---------------------------------------------------
# Moving average of batch losses
# ---------------------------------------------------
window = 20  # adjust as needed

if len(batch_losses) >= window:
    smoothed = np.convolve(
        batch_losses,
        np.ones(window) / window,
        mode="valid"
    )
    smoothed_steps = batch_steps[window - 1:]
else:
    smoothed = batch_losses
    smoothed_steps = batch_steps

# ---------------------------------------------------
# Plot batch losses
# ---------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(batch_steps, batch_losses, alpha=0.3, label="Batch Loss")
plt.plot(smoothed_steps, smoothed, linewidth=2,
         label=f"Moving Average ({window})")

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Batch Loss During Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Plot epoch average losses
# ---------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(epochs, epoch_losses, marker="o")

best_epoch = epochs[np.argmin(epoch_losses)]
best_loss = min(epoch_losses)

plt.scatter(best_epoch, best_loss,
            color="red",
            label=f"Best Epoch = {best_epoch}")

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Epoch Average Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Lowest epoch loss: {best_loss:.4f} at epoch {best_epoch}")