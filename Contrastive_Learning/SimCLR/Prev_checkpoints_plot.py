import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Setup Data Paths
real_images = sorted(glob.glob('test/real/*.*'))
aug_images = sorted(glob.glob('test/augmentations/*.*'))

if len(real_images) == 0 or len(real_images) != len(aug_images):
    print("❌ ERROR: Check your folders. Make sure both have the same number of images.")
    exit()

print(f"✅ Found {len(real_images)} image pairs.")

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs_to_test = [500, 800, 1030]

for epoch in epochs_to_test:
    checkpoint_path = f'simclr_checkpoint_epoch_{epoch}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: {checkpoint_path} not found. Skipping...")
        continue

    print(f"\n🚀 Rendering Visuals for Epoch {epoch}...")
    
    # Load Model
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Identity()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('encoder.'):
            k = k.replace('encoder.', '')
            if k.startswith('0.'): k = k.replace('0.', 'conv1.', 1)
            elif k.startswith('1.'): k = k.replace('1.', 'bn1.', 1)
            elif k.startswith('4.'): k = k.replace('4.', 'layer1.', 1)
            elif k.startswith('5.'): k = k.replace('5.', 'layer2.', 1)
            elif k.startswith('6.'): k = k.replace('6.', 'layer3.', 1)
            elif k.startswith('7.'): k = k.replace('7.', 'layer4.', 1)
            state_dict[k] = v

    resnet.load_state_dict(state_dict, strict=False)
    resnet = resnet.to(device)
    resnet.eval()

    # Extract Features
    real_vectors = []
    aug_vectors = []

    with torch.no_grad():
        for real_path, aug_path in zip(real_images, aug_images):
            img_r = Image.open(real_path).convert('RGB')
            img_a = Image.open(aug_path).convert('RGB')
            
            tr_r = base_transform(img_r).unsqueeze(0).to(device)
            tr_a = base_transform(img_a).unsqueeze(0).to(device)
            
            real_vectors.append(resnet(tr_r).squeeze().cpu())
            aug_vectors.append(resnet(tr_a).squeeze().cpu())

    # Math
    real_matrix = F.normalize(torch.stack(real_vectors), dim=1)
    aug_matrix = F.normalize(torch.stack(aug_vectors), dim=1)
    similarity_matrix = torch.matmul(real_matrix, aug_matrix.T).numpy()

    # Calculate Metrics
    n = len(real_vectors)
    diagonal_mask = np.eye(n, dtype=bool)
    
    true_pairs = similarity_matrix[diagonal_mask]
    false_pairs = similarity_matrix[~diagonal_mask]
    
    avg_true = np.mean(true_pairs)
    avg_false = np.mean(false_pairs)
    
    top1_matches = np.argmax(similarity_matrix, axis=1)
    correct = np.sum(top1_matches == np.arange(n))
    top1_acc = (correct / n) * 100

    # --- CREATE THE PRESENTATION FIGURE ---
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('white')

    # Subplot 1: The Heatmap
    ax1 = plt.subplot(1, 2, 1)
    cax = ax1.matshow(similarity_matrix, cmap='magma', vmin=-0.2, vmax=1.0)
    fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Cosine Similarity Matrix", fontsize=18, pad=20)

    # Subplot 2: The Metrics Bar Chart
    ax2 = plt.subplot(1, 2, 2)
    categories = ['Matching Pairs (True)', 'Mismatched Pairs (False)']
    values = [avg_true, avg_false]
    colors = ['#2ca02c', '#d62728'] # Green and Red
    
    bars = ax2.bar(categories, values, color=colors, width=0.5)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Average Cosine Similarity", fontsize=14)
    ax2.set_title("Signal vs. Noise Separation", fontsize=18, pad=20)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Add the exact numbers on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add massive text for Top-1 Accuracy
    fig.text(0.5, 0.85, f"Top-1 Retrieval Accuracy: {top1_acc:.1f}%", ha='center', va='center', fontsize=20, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.3, pad=10, boxstyle='round,pad=0.5'))
    
    # Super Title
    plt.suptitle(f"SimCLR Evaluation: Epoch {epoch}", fontsize=24, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout to fit the suptitle
    plt.savefig(f'epoch_{epoch}_slide.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n✅ Saved all presentation slides! Open epoch_1030_slide.png to check it out.")