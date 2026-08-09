import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ---------------------------------------------------------
# 1. THE KERNEL BANK (Bare Metal Filters)
# ---------------------------------------------------------

class RawGaussianFilter(nn.Module):
    def __init__(self, kernel_size=7, sigma=1.5):
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
    def __init__(self, kernel_size=5):
        super().__init__()
        self.padding = kernel_size // 2
        # A matrix of uniform probabilities (1 / K^2)
        b2d = torch.ones((kernel_size, kernel_size), dtype=torch.float32) / (kernel_size ** 2)
        self.register_buffer('kernel', b2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.padding, groups=3)

class RawMotionFilter(nn.Module):
    """ Models camera shake. An anisotropic diagonal kernel. """
    def __init__(self, kernel_size=7):
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
            RawGaussianFilter(kernel_size=5, sigma=1.0),
            RawGaussianFilter(kernel_size=7, sigma=2.0),
            RawBoxFilter(kernel_size=5),
            RawMotionFilter(kernel_size=7)
        ])

    def forward(self, x):
        # 1. Stochastic Blur Selection (Choose 1 generalized kernel randomly)
        selected_blur = random.choice(self.blur_bank)
        x_aug = selected_blur(x)

        # 2. Additive White Gaussian Noise (AWGN) with random intensity
        # Noise std dev randomly chosen between 0.01 and 0.05
        noise_std = random.uniform(0.01, 0.05)
        n_i = torch.randn_like(x_aug) * noise_std
        x_aug = x_aug + n_i

        # 3. Mild Global Intensity Shift (Brightness)
        # Random scalar between 0.9 (darker) and 1.1 (brighter)
        intensity_scalar = random.uniform(0.9, 1.1)
        x_aug = x_aug * intensity_scalar

        # Clamp to [0,1] — valid since TF.to_tensor() normalises images to [0,1].
        # If the dataloader ever changes to return [0,255] tensors, remove this.
        return torch.clamp(x_aug, 0.0, 1.0)

# ---------------------------------------------------------
# 3. THE HARD NEGATIVE (Malicious Forgery)
# ---------------------------------------------------------

class LocalMaliciousForgery(nn.Module):
    """
    Applies heavy localised Gaussian blur inside a random rectangle to simulate
    a realistic forgery (e.g. watermark removal, region tampering).

    Two fixes over the naive version:
      - Soft mask boundary: the binary rectangle mask is convolved with a
        Gaussian before blending, so the forgery has a feathered edge instead
        of a hard boundary. This prevents the model from learning the shortcut
        of detecting the edge discontinuity rather than the blur anomaly itself.
      - Precomputed kernels: both the heavy blur bank and the soft mask kernel
        are built once in __init__ and stored as registered buffers / ModuleList.
        The original code instantiated a new RawGaussianFilter on every forward
        call, which re-allocated and re-computed the kernel every batch.

    FIX [BUG 7]: Mask indexing calls .item() on CUDA scalar tensors before
    using them as Python slice indices. Using raw CUDA rank-0 tensors as slice
    bounds can silently fail or raise errors depending on PyTorch version.
    """
    def __init__(self, tamper_size=(64, 128)):
        super().__init__()
        self.tamper_size = tamper_size

        # Precomputed heavy blur bank — four sigma levels for variance across batches.
        # Using nn.ModuleList so .to(device) propagates correctly (same reason as BUG 3).
        # Kernel size 31 chosen for 512x512 images: covers ~6% of image width,
        # giving a visually significant but not total blur.
        self.heavy_blur_bank = nn.ModuleList([
            RawGaussianFilter(kernel_size=31, sigma=s)
            for s in [5.0, 6.0, 7.0, 8.0]
        ])

        # Precomputed soft mask kernel — built once, stored as a buffer.
        # Applied to the binary rectangle mask to feather its edges before blending.
        # sigma=7.0 on a 31x31 kernel gives ~15px soft boundary on a 512x512 image.
        _ks = 31
        _sigma = 7.0
        _pad = _ks // 2
        coords = torch.arange(-_pad, _pad + 1, dtype=torch.float32)
        g1d = torch.exp(-(coords ** 2) / (2 * _sigma ** 2))
        g2d = g1d.view(-1, 1) * g1d.view(1, -1)
        g2d = g2d / g2d.sum()
        # Shape (1, 1, 31, 31) — single channel, applied to the single-channel mask.
        self.register_buffer('soft_kernel', g2d.view(1, 1, _ks, _ks))
        self.soft_padding = _pad

    def _make_soft_mask(self, hard_mask):
        # Convolve the binary mask with the Gaussian kernel to feather the boundary.
        # soft_kernel is already on the correct device via register_buffer.
        return F.conv2d(hard_mask, self.soft_kernel, padding=self.soft_padding)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        min_box = min(self.tamper_size[0], H, W)
        max_box = min(self.tamper_size[1], H, W)
        if min_box < 1 or max_box < 1:
            raise ValueError(
                f"Input spatial size {(H, W)} is too small for tamper_size={self.tamper_size}."
            )

        # Pick one blur level randomly from the precomputed bank.
        heavy_blur = random.choice(self.heavy_blur_bank)

        # Build the binary rectangle mask — one random box per image in the batch.
        mask = torch.zeros((B, 1, H, W), device=device)

        box_h = torch.randint(min_box, max_box + 1, (B,), device=device)
        box_w = torch.randint(min_box, max_box + 1, (B,), device=device)
        # clamp(min=0) guards against box_h > H on pathologically small images.
        y_start = (torch.rand(B, device=device) * (H - box_h).clamp(min=0)).long()
        x_start = (torch.rand(B, device=device) * (W - box_w).clamp(min=0)).long()

        for i in range(B):
            # FIX [BUG 7]: .item() converts CUDA scalar tensor to Python int
            # before use as a slice index.
            y0 = y_start[i].item()
            x0 = x_start[i].item()
            h_i = box_h[i].item()
            w_i = box_w[i].item()
            mask[i, 0, y0:y0 + h_i, x0:x0 + w_i] = 1.0

        blurred_x = heavy_blur(x)

        # Feather the mask boundary, then blend blurred and original regions.
        # Inside box (soft_mask ≈ 1): blurred pixels.
        # Outside box (soft_mask ≈ 0): original pixels.
        # Boundary: smooth blend between the two.
        soft_mask = self._make_soft_mask(mask)
        return (soft_mask * blurred_x) + ((1.0 - soft_mask) * x)