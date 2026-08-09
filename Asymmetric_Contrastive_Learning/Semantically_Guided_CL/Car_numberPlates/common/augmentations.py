"""SimCLR augmentation pipeline shared by global images and semantic patches."""

from __future__ import annotations

from torchvision import transforms

from common.preprocessing import ResizeLongSideAndPad


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_simclr_transform(image_size: int = 224) -> transforms.Compose:
    """Build the standard SimCLR augmentation pipeline.

    The first operation performs aspect-ratio preserving preprocessing. All
    following operations are standard SimCLR augmentations and are intentionally
    identical for full images and cropped semantic patches.
    """

    color_jitter = transforms.ColorJitter(
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2,
    )
    return transforms.Compose(
        [
            ResizeLongSideAndPad(image_size),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=max(3, int(0.1 * image_size) // 2 * 2 + 1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

