"""Aspect-preserving image preprocessing used by both experiments."""

from __future__ import annotations

from PIL import Image, ImageOps


class ResizeLongSideAndPad:
    """Resize so the long side equals ``size`` and pad to ``size x size``.

    This avoids geometric distortion while producing tensors with a fixed
    spatial size. Padding is symmetric except for a one-pixel difference when
    the resized side has odd parity.
    """

    def __init__(self, size: int = 224, fill: int | tuple[int, int, int] = 0) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Image has invalid dimensions.")

        scale = self.size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)

        pad_width = self.size - resized_width
        pad_height = self.size - resized_height
        left = pad_width // 2
        top = pad_height // 2
        right = pad_width - left
        bottom = pad_height - top
        return ImageOps.expand(resized, border=(left, top, right, bottom), fill=self.fill)

