import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image

# 1. Define Transforms (Keeping the safe 60% crop)
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=9),
    transforms.ToTensor(),
])

# 2. Load the Model with Translation Block
print("Loading SimCLR checkpoint 1030...")
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

# PUSH TO GPU for lightning speed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Moving model to {device}...")
resnet = resnet.to(device)
resnet.eval()

# 3. Grab exactly 200 images from UTK_Face/part1
image_files = glob.glob('UTK_Face/part1/*.jpg') + glob.glob('UTK_Face/part1/*.png')
image_files = sorted(image_files)[:200]

if len(image_files) == 0:
    print("❌ ERROR: Could not find any images in UTK_Face/part1/ ! Check your path.")
    exit()

print(f"✅ Found {len(image_files)} images. Generating 200x200 Cosine Similarity Heatmap...")

actual_vectors = []
aug_vectors = []

with torch.no_grad():
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Transform and push to GPU
            img_act = base_transform(img).unsqueeze(0).to(device)
            img_aug = aug_transform(img).unsqueeze(0).to(device)
            
            # Forward pass, then instantly pull back to CPU so we don't hold VRAM
            actual_vectors.append(resnet(img_act).squeeze().cpu())
            aug_vectors.append(resnet(img_aug).squeeze().cpu())
        except Exception as e:
            print(f"Skipping corrupted image {img_path}: {e}")

# 4. Math: Stack and Normalize
actual_matrix = F.normalize(torch.stack(actual_vectors), dim=1)
aug_matrix = F.normalize(torch.stack(aug_vectors), dim=1)

# Compute Cosine Similarity Matrix
similarity_matrix = torch.matmul(actual_matrix, aug_matrix.T).numpy()

# 5. Plotting the Grid
fig, ax = plt.subplots(figsize=(16, 16))

# Generate the heatmap
cax = ax.matshow(similarity_matrix, cmap='magma', vmin=-0.2, vmax=1.0)
fig.colorbar(cax, fraction=0.046, pad=0.04)

# Turn off the tick labels entirely. 200 text labels would just look like a solid black barcode.
ax.set_xticks([])
ax.set_yticks([])

plt.title("SimCLR Domain Check: UTK_Face (200 Images)", fontsize=22, pad=20)

plt.tight_layout()
plt.savefig('heatmap_utk_200.png', dpi=300)
print("✅ Saved UTK validation heatmap as heatmap_utk_200.png!")