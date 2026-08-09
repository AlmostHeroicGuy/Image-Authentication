import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ---------------------------------------------------------
# 1. THE KERNEL BANK (Bare Metal Filters)
# ---------------------------------------------------------

class RawGaussianFilter(nn.Module):
    def __init__(self, kernel_size, sigma):
        super().__init__()
        self.padding = kernel_size // 2
        # Here padding is done to make sure that the size of the output image is the same as the input as convolution shrinks the image.
        # The general formula for an output image size after convolution is: [ input_size + 2*padding - kernel_size ] / stride + 1
        # For a stride of 1, the padding for same sized output as the input is: [ padding = kernel_size - 1] / 2
        coords = torch.arange(-self.padding, self.padding + 1, dtype=torch.float32)
        # These coordinates are to make a 1D gaussian function across the centre of pixel of where we are applying the convolution right now
        # for a kernel size of 7, we make a 7 element array to make 7 outputs of the gaussian function across the centre [-3, -2, -1, 0, 1 , 2, 3]
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        # This is the 1D gaussian function in action
        g2d = g1d.view(-1, 1) * g1d.view(1, -1)
        # Here we changed the tensor shape from (7,) to (7, 1) by using the grid.view(-1, 1) and similar logic for the other tensor that we made
        # Then we multiply the column vector with the row vector to get a 2D gaussian kernel (matrix) of shape (7, 7)
        g2d = g2d / torch.sum(g2d)
        # Finally, we normalize the kernel so that the sum of all its elements is 1. This ensures that the overall brightness of the image is preserved after convolution.
        self.register_buffer('kernel', g2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1))
        # buffer is just to put this as a non-trainable parameter of the module, so that it gets moved to the correct device (CPU/GPU) along with the rest of the model when we call .to(device) on the parent module.
        # Now we have the 2D Gaussian kernel of shape (7, 7) and we reshape it to (1, 1, 7, 7) to make it compatible for convolution with a batch of images.
        # The repeat(3, 1, 1, 1) is used to create a separate kernel for each of the 3 color channels (R, G, B) in the input image. This way, we can apply the same Gaussian blur to each channel independently during convolution.
        # repeat(a, b, c, d) will repeat the tensor a times along the first dimension, b times along the second dimension, and so on. So in this case, we are repeating the kernel 3 times along the first dimension to create 3 separate kernels for the 3 color channels.

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding, groups=3)

class RawBoxFilter(nn.Module):
    """ Models cheap camera sensor integration or naive downsampling. """
    def __init__(self, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        # A matrix of uniform probabilities (1 / K^2)
        b2d = torch.ones((kernel_size, kernel_size), dtype=torch.float32) / (kernel_size ** 2)
        self.register_buffer('kernel', b2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding, groups=3)

class RawMotionFilter(nn.Module):
    """ Models camera shake. An anisotropic diagonal kernel. """
    def __init__(self, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        # Create an identity matrix (diagonal line of 1s) and normalize
        m2d = torch.eye(kernel_size, dtype=torch.float32) / kernel_size
        self.register_buffer('kernel', m2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding, groups=3)

# ---------------------------------------------------------
# 2. THE GLOBAL MIXTURE (Benign Degradation)
# ---------------------------------------------------------

class GlobalBenignAugmentation(nn.Module):
    """
    Applies a stochastic composition of generalized blurs, noise, and intensity shifts.

    FIX [BUG 3]: blur_bank changed from a plain Python list to nn.ModuleList.
    A plain list is invisible to PyTorch's module system — calling .to(device)
    on the parent module would NOT move the filter submodules to the GPU.
    The forward pass would then fail with a device mismatch error when a CUDA
    image tensor hits a CPU-resident convolution kernel.
    nn.ModuleList registers the submodules properly, so .to(), .cuda(),
    .parameters(), and .state_dict() all propagate correctly.
    """
    def __init__(self):
        super().__init__()
        # FIX [BUG 3]: nn.ModuleList instead of plain list.
        self.blur_bank = nn.ModuleList([
            RawGaussianFilter(kernel_size=3, sigma=0.8),
            RawBoxFilter(kernel_size=3),
            RawMotionFilter(kernel_size=3)
        ])

    def forward(self, x):
        # 1. Stochastic Blur Selection (Choose 1 generalized kernel randomly)
        idx = torch.randint(len(self.blur_bank), (1,), device=x.device).item()
        selected_blur = self.blur_bank[idx]
        x_aug = selected_blur(x)

        # 2. Additive White Gaussian Noise (AWGN) with random intensity
        # Noise std dev randomly chosen between 0.01 and 0.05
        noise_std = torch.empty(1, device=x.device).uniform_(0.01, 0.05).item()
        n_i = torch.randn_like(x_aug) * noise_std
        x_aug = x_aug + n_i

        # 3. Mild Global Intensity Shift (Brightness)
        # Random scalar between 0.9 (darker) and 1.1 (brighter)
        intensity_scalar = torch.empty(1, device=x.device).uniform_(0.9, 1.1).item()
        x_aug = x_aug * intensity_scalar

        # Clamp to [0,1] — valid since TF.to_tensor() normalises images to [0,1].
        # If the dataloader ever changes to return [0,255] tensors, remove this.
        return torch.clamp(x_aug, 0.0, 1.0)

# ---------------------------------------------------------
# 3. THE HARD NEGATIVE (Malicious Forgery)
# ---------------------------------------------------------
class LocalWatermarkForgery(nn.Module):
    """
    Inserts a fixed 8x8 semi-transparent watermark at a random location
    per sample, with Gaussian-weighted falloff from the patch center toward
    its edges. Returns (output, soft_mask); soft_mask (B,1,H,W) can serve as
    a ground-truth localization map for this forgery.
    """
    def __init__(self, value_range=(0.0, 1.0)):
        super().__init__()
        pattern = torch.tensor([
            [1,1,0,0,1,1,0,0],
            [1,1,0,0,1,1,0,0],
            [0,0,1,1,0,0,1,1],
            [0,0,1,1,0,0,1,1],
            [1,1,0,0,1,1,0,0],
            [1,1,0,0,1,1,0,0],
            [0,0,1,1,0,0,1,1],
            [0,0,1,1,0,0,1,1],
        ], dtype=torch.float32)

        # Remap {0,1} pattern into whatever value range your images live in
        # post-normalization (e.g. pass each channel's (min,max) if you
        # normalize with mean/std, so the watermark is actually visible
        # rather than just an arbitrary perturbation).
        low, high = value_range
        pattern = low + pattern * (high - low)

        watermark = pattern.unsqueeze(0).repeat(3, 1, 1)  # (3,8,8)
        self.register_buffer("watermark", watermark.unsqueeze(0))  # (1,3,8,8)

        ks, sigma, pad = 11, 2.0, 5  # ks= kernel size, sigma=std dev of gaussian, pad=padding for conv2d
        coords = torch.arange(-pad, pad + 1, dtype=torch.float32)
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g2d = g1d[:, None] * g1d[None, :]
        g2d /= g2d.sum()
        self.register_buffer("soft_kernel", g2d.view(1, 1, ks, ks))
        self.padding = pad

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        
        y0 = torch.randint(0, H - 7, (B,), device=device)   # (B,)
        x0 = torch.randint(0, W - 7, (B,), device=device)   # (B,)
        alpha = torch.empty(B, device=device).uniform_(0.25, 0.40)  # (B,)

        rows = torch.arange(H, device=device).view(1, H, 1)  # (1,H,1)
        cols = torch.arange(W, device=device).view(1, 1, W)  # (1,1,W)
        y0e, x0e = y0.view(B,1,1), x0.view(B,1,1)

        in_box = (rows >= y0e) & (rows < y0e+8) & (cols >= x0e) & (cols < x0e+8)  # (B,H,W)
        mask = in_box.unsqueeze(1).float()                 # (B,1,H,W)
        alpha_mask = mask * alpha.view(B,1,1,1)            # (B,1,H,W)

        # gather the right 8x8 watermark value for every pixel, per-sample offset
        rel_h = (rows - y0e).clamp(0, 7)   # (B,H,W)
        rel_w = (cols - x0e).clamp(0, 7)   # (B,H,W)
        flat_idx = (rel_h * 8 + rel_w).reshape(B, -1)        # (B, H*W)
        wm_flat = self.watermark[0].reshape(3, 64)           # (3,64)
        wm_canvas = wm_flat[:, flat_idx].reshape(3, B, H, W).permute(1,0,2,3) * mask

        # Single batched conv2d instead of B separate calls
        soft_mask = F.conv2d(mask, self.soft_kernel, padding=self.padding)

        output = x + soft_mask * alpha_mask * (wm_canvas - x)
        return output, soft_mask