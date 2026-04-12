import os
import glob
import shutil
import torchvision.transforms as transforms
from PIL import Image

# 1. Setup Directories
real_dir = 'test/real'
aug_dir = 'test/augmentations'

os.makedirs(real_dir, exist_ok=True)
os.makedirs(aug_dir, exist_ok=True)

# 2. Grab 200 images from UTK_Face part1
source_images = glob.glob('UTK_Face/part1/*.jpg') + glob.glob('UTK_Face/part1/*.png')
source_images = sorted(source_images)[:200]

if not source_images:
    print("❌ ERROR: Could not find images in UTK_Face/part1/")
    exit()

# 3. Define the SimCLR Augmentation Pipeline (No ToTensor, we want to save them as images)
color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([color_jitter], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # Using a standard PIL blur since Torchvision's GaussianBlur acts weird on raw PIL images sometimes
    transforms.GaussianBlur(kernel_size=9) 
])

# To ensure images are at least resized correctly for the 'real' folder
resize_only = transforms.Resize((224, 224))

print(f"Processing 200 images into {real_dir} and {aug_dir}...")

for img_path in source_images:
    filename = os.path.basename(img_path)
    
    try:
        # Load the original image
        img = Image.open(img_path).convert('RGB')
        
        # Resize original and save to real folder
        real_img = resize_only(img)
        real_img.save(os.path.join(real_dir, filename))
        
        # Apply SimCLR augmentations and save to aug folder
        aug_img = aug_transform(img)
        aug_img.save(os.path.join(aug_dir, filename))
        
    except Exception as e:
        print(f"Skipped {filename} due to error: {e}")

print("✅ Done! Images are ready in test/real and test/augmentations.")