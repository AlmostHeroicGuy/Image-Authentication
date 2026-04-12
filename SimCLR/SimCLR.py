import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class SimCLR(nn.Module):
    def __init__(self, projection_dim=128):
        super(SimCLR, self).__init__()
        
        # 1. Base Encoder: f(.)
        # We load a ResNet-50. You can start from scratch (weights=None) 
        # or use ImageNet weights to speed up convergence on your deepfake dataset.
        base_model = resnet50(weights=None)
        
        # The output of ResNet-50 before the FC layer is 2048 dimensions
        self.feature_dim = base_model.fc.in_features
        
        # We strip off the final fully connected (classification) layer
        # list(base_model.children())[:-1] takes everything EXCEPT the last layer
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        
        # 2. Projection Head: g(.)
        # The paper uses an MLP with one hidden layer and a ReLU non-linearity.
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim)
        )

    def forward(self, x):
        # Extract the representation h
        h = self.encoder(x)
        
        # Flatten the spatial dimensions (batch_size, 2048, 1, 1) -> (batch_size, 2048)
        h = torch.flatten(h, start_dim=1)
        
        # Project it to the latent space z
        z = self.projection_head(h)
        
        # We return both. During pretraining, we use 'z' for the contrastive loss.
        # After pretraining, we throw away the projection head and only use 'h'.
        return h, z

# ==========================================
# SANITY CHECK (Run this to verify shapes)
# ==========================================
if __name__ == "__main__":
    # Create the model
    model = SimCLR(projection_dim=128)
    
    # Create a dummy batch of 4 images, 3 channels, 224x224 pixels
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Pass it through the model
    h, z = model(dummy_input)
    
    print(f"Representation 'h' shape: {h.shape}") # Should be [4, 2048]
    print(f"Projection 'z' shape: {z.shape}")     # Should be [4, 128]