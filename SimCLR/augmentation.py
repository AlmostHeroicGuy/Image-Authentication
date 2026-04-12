import torch
from torchvision.transforms import v2

class AddGaussianNoise(torch.nn.Module):
    """Custom module to inject random Gaussian noise for deepfake forensics"""
    def __init__(self, p=0.5, mean=0., std=0.05):
        super().__init__()
        self.p = p
        self.mean = mean
        self.std = std

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(img.size()) * self.std + self.mean
            # Add noise and clamp values between 0 and 1
            img = torch.clamp(img + noise.to(img.device), 0.0, 1.0)
        return img

class SimCLRAugmentation:
    """
    A stochastic data augmentation module that transforms any given data example 
    randomly resulting in two correlated views of the same example.
    """
    def __init__(self, image_size=224):
        # We define the stochastic pipeline once
        self.transform = v2.Compose([
            # 1. Convert PIL image to uint8 Tensor (Required for JPEG transform)
            v2.ToImage(), 
            
            # 2. Random Crop and Resize (Crucial for SimCLR)
            v2.RandomResizedCrop(size=image_size, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            
            # 3. Horizontal Flip (50% probability)
            v2.RandomHorizontalFlip(p=0.5),
            
            # 4. Color Distortion (Jitter applied 80% of the time)
            v2.RandomApply([
                v2.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ], p=0.8),
            
            # 5. Color Dropping (Grayscale applied 20% of the time)
            v2.RandomGrayscale(p=0.2),
            
            # 6. Gaussian Blur (Applied 50% of the time)
            v2.RandomApply([
                v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            
            # 7. JPEG Compression (Applied 50% of the time, random quality 10-100)
            v2.RandomApply([
                v2.JPEG(quality=(10, 100))
            ], p=0.5),
            
            # 8. Convert to float32 [0.0, 1.0] for the neural network
            v2.ToDtype(torch.float32, scale=True),
            
            # 9. Gaussian Noise (Custom for deepfakes, 50% probability)
            AddGaussianNoise(p=0.5, std=0.05),
            
            # 10. Cutout / Random Erasing (50% probability)
            v2.RandomErasing(p=0.5, scale=(0.02, 0.2), value=0) 
        ])

    def __call__(self, x):
        """
        When this class is called on an image, it passes the image through 
        the stochastic pipeline twice to generate two different random views.
        """
        view_1 = self.transform(x)
        view_2 = self.transform(x)
        return view_1, view_2

# ==========================================
# HOW TO PLUG THIS INTO YOUR PYTORCH DATASET
# ==========================================
# In your main script, you just pass this class as the transform argument
# to your Dataset. PyTorch's DataLoader will handle the rest!

# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader

# custom_transform = SimCLRAugmentation(image_size=224)
# dataset = ImageFolder(root="/home/tushar/SimCLR/archive (1)", transform=custom_transform)
# dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)