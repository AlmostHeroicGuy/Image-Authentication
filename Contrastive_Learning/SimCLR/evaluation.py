import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# 1. Load the Base ResNet-50 Architecture
# We use False because we want YOUR weights, not ImageNet's
resnet = models.resnet50(weights=None)
resnet.fc = nn.Identity()  # Strip the default classification head

# 2. Load Your 1030-Epoch Checkpoint
checkpoint_path = 'simclr_checkpoint_epoch_1030.pth'
checkpoint = torch.load(checkpoint_path)

# Extract just the ResNet backbone weights (ignoring the projection head)
state_dict = checkpoint['model_state_dict']
backbone_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('backbone.'): # Adjust this string if you named it differently in SimCLR.py
        # Remove 'backbone.' prefix so it matches standard ResNet keys
        new_key = key.replace('backbone.', '')
        backbone_state_dict[new_key] = value

# Load the weights strictly into the backbone
resnet.load_state_dict(backbone_state_dict, strict=False)
print("✅ Checkpoint 1030 loaded successfully.")

# 3. FREEZE THE BACKBONE (Crucial Step!)
for param in resnet.parameters():
    param.requires_grad = False

# 4. Build the Linear Classifier Model
class SimCLRLinearClassifier(nn.Module):
    def __init__(self, frozen_backbone, num_classes=2):
        super().__init__()
        self.backbone = frozen_backbone
        # The only layer that will actually learn
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        with torch.no_grad(): # Double-lock to ensure ResNet doesn't update
            features = self.backbone(x)
        return self.classifier(features)

model = SimCLRLinearClassifier(frozen_backbone=resnet)
model = model.cuda()

# 5. Setup Optimizer and Loss
# We ONLY pass the classifier's parameters to the optimizer!
optimizer = optim.Adam(model.classifier.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

print("🚀 Model is ready for Linear Evaluation. Only the final layer will be trained.")

# --- YOUR DATALOADER GOES HERE ---
# You will need your dataset with ACTUAL labels now (0 for Real, 1 for Fake).
# You do NOT need the heavy SimCLR augmentations here. Just Resize, CenterCrop, and Normalize.