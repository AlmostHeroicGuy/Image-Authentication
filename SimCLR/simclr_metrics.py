import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# 1. Setup Data Paths
# Sorting ensures real_images[0] corresponds exactly to aug_images[0]
real_images = sorted(glob.glob('test/real/*.*'))
aug_images = sorted(glob.glob('test/augmentations/*.*'))

if len(real_images) == 0 or len(real_images) != len(aug_images):
    print("❌ ERROR: Check your folders. Make sure both have the same number of images.")
    exit()

print(f"✅ Found {len(real_images)} image pairs. Loading Model...")

# 2. Load the SimCLR Model (With Translation Block)
resnet = models.resnet50(weights=None)
resnet.fc = torch.nn.Identity()
checkpoint = torch.load('simclr_checkpoint_epoch_1030.pth', map_location='cpu')

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)
resnet.eval()

# 3. Extract Features
# We just need to resize and tensorify since the augmentations are already saved
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

real_vectors = []
aug_vectors = []

print("Extracting features (this will be fast on GPU)...")
with torch.no_grad():
    for real_path, aug_path in zip(real_images, aug_images):
        try:
            img_r = Image.open(real_path).convert('RGB')
            img_a = Image.open(aug_path).convert('RGB')
            
            tr_r = base_transform(img_r).unsqueeze(0).to(device)
            tr_a = base_transform(img_a).unsqueeze(0).to(device)
            
            real_vectors.append(resnet(tr_r).squeeze().cpu())
            aug_vectors.append(resnet(tr_a).squeeze().cpu())
        except Exception as e:
            print(f"Skipping pair {real_path} due to error: {e}")

# 4. Compute Similarity Matrix
real_matrix = F.normalize(torch.stack(real_vectors), dim=1)
aug_matrix = F.normalize(torch.stack(aug_vectors), dim=1)

# similarity_matrix[i, j] = cosine similarity between real_i and aug_j
similarity_matrix = torch.matmul(real_matrix, aug_matrix.T).numpy()

# 5. Calculate Metrics
n = len(real_vectors)

# Create a boolean mask for the diagonal (True Pairs)
diagonal_mask = np.eye(n, dtype=bool)

true_pairs = similarity_matrix[diagonal_mask]
false_pairs = similarity_matrix[~diagonal_mask]

avg_true_sim = np.mean(true_pairs)
avg_false_sim = np.mean(false_pairs)

# Top-1 Retrieval Accuracy
# For each row (real image), find the index of the highest similarity column (aug image)
top1_matches = np.argmax(similarity_matrix, axis=1)
# Check how many times the highest similarity index matches the actual correct index
correct_matches = np.sum(top1_matches == np.arange(n))
top1_accuracy = (correct_matches / n) * 100

# 6. Print the Demonstration Report
print("\n" + "="*50)
print("🚀 SIMCLR CHECKPOINT 1030 EVALUATION REPORT")
print("="*50)
print(f"Total Pairs Tested:      {n}")
print("-" * 50)
print(f"✅ Average TRUE Pair Similarity:   {avg_true_sim:.4f}")
print(f"❌ Average FALSE Pair Similarity:  {avg_false_sim:.4f}")
print(f"📊 Signal-to-Noise Margin:         {(avg_true_sim - avg_false_sim):.4f}")
print("-" * 50)
print(f"🎯 Top-1 Retrieval Accuracy:       {top1_accuracy:.2f}% ({correct_matches}/{n} correct)")
print("="*50 + "\n")