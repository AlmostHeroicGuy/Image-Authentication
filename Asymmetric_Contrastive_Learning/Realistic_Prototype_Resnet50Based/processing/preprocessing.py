"""Shared image preprocessing for training and evaluation."""

import torchvision.transforms.functional as TF


def preprocess_image(image, image_size: int):
    """Resize the shorter side, center-crop a square, and return a [0, 1] tensor."""
    image = TF.resize(image, image_size, antialias=True)
    image = TF.center_crop(image, [image_size, image_size])
    return TF.to_tensor(image)
