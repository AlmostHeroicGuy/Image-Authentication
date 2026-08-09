import torch
import torch.nn as nn

class SpatialTokenizer(nn.Module):
    """
    Module B: Spatial-to-Sequence Tokenizer
    Bridges the Convolutional backbone to the Transformer head by flattening 
    the spatial grid, projecting channel dimensions, and injecting absolute 
    positional embeddings to break permutation invariance.
    """
    def __init__(self, in_channels=1024, spatial_size=32, hidden_dim=768):
        super(SpatialTokenizer, self).__init__()
        
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        
        # Calculate total tokens mathematically: N = H * W -> 32 * 32 = 1024 patches
        self.num_patches = spatial_size * spatial_size
        
        # 1. Linear Projection (W_p): Maps CNN channels to Transformer Hidden Dim
        # Squeezes R^1024 -> R^768
        self.patch_proj = nn.Linear(in_channels, hidden_dim)
        
        # 2. The [CLS] Token (z_cls): Learnable global anomaly aggregator
        # Shape: [1, 1, 768]
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 3. Absolute Positional Embeddings (E_pos): Spatial coordinate map
        # We need num_patches (1024) + 1 coordinates to account for the [CLS] token
        # Shape: [1, 1025, 768]
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        
        # Initialize weights for optimization stability
        self._init_weights()

    def _init_weights(self):
        """
        Truncated normal initialization prevents extreme outlier weights in the 
        embeddings, which can saturate the Softmax function in early training epochs.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)

    def forward(self, F_map):
        """
        Forward Pass Signal Flow:
        F_map: [Batch, 1024 (Channels), 32 (Height), 32 (Width)]
        Returns Z_0: [Batch, 1025 (Tokens), 768 (Hidden Dim)]
        """
        B = F_map.shape[0]
        
        # --- Step 1: Spatial Flattening ---
        # .view() flattens the 32x32 grid into 1024 linear patches.
        # .permute() swaps the axes to align with Transformer sequence expectations (Batch, Sequence, Channels)
        # Transform: [B, 1024, 32, 32] -> [B, 1024, 1024 patches] -> [B, 1024 patches, 1024 channels]
        F_flat = F_map.view(B, self.in_channels, -1).permute(0, 2, 1)
        
        # --- Step 2: Subspace Projection ---
        # Project the high-frequency CNN features into the dense 768-D semantic space
        # Transform: [B, 1024 patches, 1024 channels] -> [B, 1024 patches, 768 features]
        Z_patches = self.patch_proj(F_flat)
        
        # --- Step 3: [CLS] Token Injection ---
        # Expand the single learnable [CLS] token across the entire batch
        # Transform: [1, 1, 768] -> [B, 1, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenate the [CLS] token strictly at the 0th index of the sequence
        # Transform: [B, 1, 768] cat [B, 1024, 768] -> [B, 1025, 768]
        Z_seq = torch.cat((cls_tokens, Z_patches), dim=1)
        
        # --- Step 4: Positional Embedding Injection ---
        # Element-wise addition of spatial coordinates to break permutation invariance.
        # PyTorch automatically broadcasts the [1, 1025, 768] tensor across the Batch dimension.
        Z_0 = Z_seq + self.pos_embed
        
        return Z_0