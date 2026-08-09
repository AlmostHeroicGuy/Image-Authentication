"""GPU-side benign and local-forgery augmentations for images in [0, 1]."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Fixed blur filters
# -----------------------------------------------------------------------------

class RawGaussianFilter(nn.Module):
    """Depthwise Gaussian blur with a fixed kernel."""

    def __init__(self, kernel_size: int, sigma: float):
        super().__init__()
        padding = kernel_size // 2
        coords = torch.arange(-padding, padding + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(coords.square()) / (2 * sigma**2))
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d /= kernel_2d.sum()

        self.padding = padding
        self.register_buffer(
            "kernel",
            kernel_2d.view(1, 1, kernel_size, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = self.kernel.expand(channels, -1, -1, -1)
        return F.conv2d(x, kernel, padding=self.padding, groups=channels)


class RawBoxFilter(nn.Module):
    """Depthwise uniform blur approximating simple sensor integration."""

    def __init__(self, kernel_size: int):
        super().__init__()
        kernel_2d = torch.full(
            (kernel_size, kernel_size),
            fill_value=1.0 / kernel_size**2,
            dtype=torch.float32,
        )
        self.padding = kernel_size // 2
        self.register_buffer(
            "kernel",
            kernel_2d.view(1, 1, kernel_size, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = self.kernel.expand(channels, -1, -1, -1)
        return F.conv2d(x, kernel, padding=self.padding, groups=channels)


class RawMotionFilter(nn.Module):
    """Depthwise linear motion blur in one of four simple directions."""

    def __init__(self, kernel_size: int, direction: str):
        super().__init__()
        kernel_2d = torch.zeros(kernel_size, kernel_size, dtype=torch.float32)

        if direction == "horizontal":
            kernel_2d[kernel_size // 2, :] = 1
        elif direction == "vertical":
            kernel_2d[:, kernel_size // 2] = 1
        elif direction == "diagonal":
            kernel_2d.fill_diagonal_(1)
        elif direction == "anti_diagonal":
            kernel_2d = torch.eye(kernel_size, dtype=torch.float32).flip(1)
        else:
            raise ValueError(f"Unsupported motion direction: {direction}")

        kernel_2d /= kernel_2d.sum()
        self.padding = kernel_size // 2
        self.register_buffer(
            "kernel",
            kernel_2d.view(1, 1, kernel_size, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = self.kernel.expand(channels, -1, -1, -1)
        return F.conv2d(x, kernel, padding=self.padding, groups=channels)


# -----------------------------------------------------------------------------
# Global benign degradation
# -----------------------------------------------------------------------------

class GlobalBenignAugmentation(nn.Module):
    """
    Apply one randomly selected mild blur, sensor noise, and brightness shift.

    Kernel sizes are scaled for 224-256 pixel inputs. The bank deliberately
    stops at moderate settings so the positive view remains recognizable.
    """

    def __init__(self):
        super().__init__()

        gaussian_blurs = [
            RawGaussianFilter(kernel_size=5, sigma=1.0),
            RawGaussianFilter(kernel_size=7, sigma=1.4),
            RawGaussianFilter(kernel_size=9, sigma=1.8),
            RawGaussianFilter(kernel_size=11, sigma=2.2),
        ]
        box_blurs = [
            RawBoxFilter(kernel_size=5),
            RawBoxFilter(kernel_size=7),
            RawBoxFilter(kernel_size=9),
        ]
        motion_blurs = [
            RawMotionFilter(kernel_size=5, direction="horizontal"),
            RawMotionFilter(kernel_size=7, direction="vertical"),
            RawMotionFilter(kernel_size=9, direction="diagonal"),
            RawMotionFilter(kernel_size=11, direction="anti_diagonal"),
        ]
        self.blur_bank = nn.ModuleList(
            gaussian_blurs + box_blurs + motion_blurs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blur_index = torch.randint(
            len(self.blur_bank), (1,), device=x.device
        ).item()
        x_aug = self.blur_bank[blur_index](x)

        noise_std = torch.empty((), device=x.device).uniform_(0.01, 0.05)
        x_aug = x_aug + torch.randn_like(x_aug) * noise_std

        brightness = torch.empty((), device=x.device).uniform_(0.9, 1.1)
        x_aug = x_aug * brightness

        return x_aug.clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Local hard negative
# -----------------------------------------------------------------------------

class LocalWatermarkForgery(nn.Module):
    """
    Blend one of three fixed 30x30 textures into a random location per image.

    The textures are smooth and non-periodic rather than high-contrast regular
    checkerboards. Their internal patterns remain unchanged; only the selected
    texture, location, and opacity vary. The returned soft mask has shape
    [B, 1, H, W] and can also be used as a localization target.
    """

    def __init__(self, patch_size: int = 30, value_range=(0.0, 1.0)):
        super().__init__()
        if patch_size < 3:
            raise ValueError("patch_size must be at least 3.")

        self.patch_size = patch_size
        low, high = value_range

        # A private generator makes the three fixed textures reproducible without
        # changing the application's global random state.
        generator = torch.Generator().manual_seed(2025)
        texture = torch.rand(
            3, 3, patch_size, patch_size, generator=generator
        )
        texture = F.avg_pool2d(texture, kernel_size=5, stride=1, padding=2)
        channel_min = texture.amin(dim=(-2, -1), keepdim=True)
        channel_max = texture.amax(dim=(-2, -1), keepdim=True)
        texture = (texture - channel_min) / (channel_max - channel_min).clamp_min(1e-6)
        texture = low + texture * (high - low)
        self.register_buffer("watermark_bank", texture)

        # Feathering softens the patch boundary while preserving a useful
        # localization mask. The kernel remains well below the patch size.
        kernel_size, sigma = 15, 3.5
        padding = kernel_size // 2
        coords = torch.arange(-padding, padding + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-(coords.square()) / (2 * sigma**2))
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d /= kernel_2d.sum()
        self.register_buffer(
            "soft_kernel", kernel_2d.view(1, 1, kernel_size, kernel_size)
        )
        self.padding = padding

    def forward(self, x: torch.Tensor):
        batch_size, channels, height, width = x.shape
        patch_size = self.patch_size

        if channels < 1:
            raise ValueError("Input must contain at least one channel.")
        if height < patch_size or width < patch_size:
            raise ValueError(
                f"Input spatial size {height}x{width} is smaller than the "
                f"{patch_size}x{patch_size} watermark."
            )

        device = x.device
        y0 = torch.randint(0, height - patch_size + 1, (batch_size,), device=device)
        x0 = torch.randint(0, width - patch_size + 1, (batch_size,), device=device)
        alpha = torch.empty(batch_size, device=device).uniform_(0.25, 0.40)
        watermark_indices = torch.randint(
            self.watermark_bank.shape[0], (batch_size,), device=device
        )

        rows = torch.arange(height, device=device).view(1, height, 1)
        cols = torch.arange(width, device=device).view(1, 1, width)
        y0_grid = y0.view(batch_size, 1, 1)
        x0_grid = x0.view(batch_size, 1, 1)

        in_patch = (
            (rows >= y0_grid)
            & (rows < y0_grid + patch_size)
            & (cols >= x0_grid)
            & (cols < x0_grid + patch_size)
        )
        hard_mask = in_patch.unsqueeze(1).to(dtype=x.dtype)

        rel_y = (rows - y0_grid).clamp(0, patch_size - 1)
        rel_x = (cols - x0_grid).clamp(0, patch_size - 1)
        flat_indices = (rel_y * patch_size + rel_x).reshape(batch_size, -1)

        # Reuse or average the fixed RGB textures as needed. This supports
        # grayscale, two-channel, RGB, RGBA, and other channel layouts without
        # making assumptions about their semantic meaning.
        if channels == 1:
            watermark_bank = self.watermark_bank.mean(dim=1, keepdim=True)
        else:
            source_channels = self.watermark_bank.shape[1]
            repeats = (channels + source_channels - 1) // source_channels
            watermark_bank = self.watermark_bank.repeat(1, repeats, 1, 1)
            watermark_bank = watermark_bank[:, :channels]

        selected_watermarks = watermark_bank[watermark_indices]
        flat_watermarks = selected_watermarks.reshape(
            batch_size, channels, patch_size**2
        )
        gather_indices = flat_indices.unsqueeze(1).expand(-1, channels, -1)
        watermark_canvas = torch.gather(
            flat_watermarks, dim=2, index=gather_indices
        ).reshape(batch_size, channels, height, width)

        soft_mask = F.conv2d(
            hard_mask, self.soft_kernel, padding=self.padding
        ) * hard_mask
        blend = soft_mask * alpha.view(batch_size, 1, 1, 1)
        output = x + blend * (watermark_canvas - x)

        return output.clamp(0.0, 1.0), soft_mask
