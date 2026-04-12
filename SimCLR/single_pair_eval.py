import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# 1. Get the filename from the command line
if len(sys.argv) < 2:
    print("❌ ERROR: You need to specify an image filename.")
    print("Usage: python3 single_pair_eval.py <image_filename>")
    print("Example: python3 single_pair_eval.py 1_0_2_20161219140530307.jpg")
    exit()

filename = sys.argv[1]

real_path = os.path.join('test/real', filename)
aug_path = os.path.join('test/augmentations', filename)

if not os.path.exists(real_path) or not os.path.exists(aug_path):
    print(f"❌ ERROR: Could not find {filename} in BOTH test/real/ and test/augmentations/")
    exit()

# 2. Load the SimCLR Model (Checkpoint 1030)
print(f"\n🧠 Loading SimCLR Checkpoint 1030...")
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

# 3. Process the Images
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print(f"📸 Loading pairs for: {filename}")
try:
    img_r = Image.open(real_path).convert('RGB')
    img_a = Image.open(aug_path).convert('RGB')
    
    tr_r = base_transform(img_r).unsqueeze(0).to(device)
    tr_a = base_transform(img_a).unsqueeze(0).to(device)
except Exception as e:
    print(f"❌ ERROR reading images: {e}")
    exit()

# 4. Extract Features and Calculate Similarity
print("⚙️ Extracting 2048-D structural representations...")
with torch.no_grad():
    vec_r = resnet(tr_r).squeeze()
    vec_a = resnet(tr_a).squeeze()

# Cosine similarity expects batches, so we add an empty dimension with unsqueeze
similarity = F.cosine_similarity(vec_r.unsqueeze(0), vec_a.unsqueeze(0)).item()

# 5. Print the Output
print("\n" + "="*50)
print(f"🎯 SINGLE PAIR EVALUATION RESULT")
print("="*50)
print(f"File: {filename}")
print(f"Real Vector Shape:       {vec_r.shape}")
print(f"Augmented Vector Shape:  {vec_a.shape}")
print("-" * 50)
print(f"✅ Cosine Similarity:     {similarity:.4f}")
print("="*50 + "\n")