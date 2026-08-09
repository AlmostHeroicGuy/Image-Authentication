"""ResNet-50 encoder for 224x224 or 256x256 RGB images."""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50(nn.Module):
    """
    ImageNet-pretrained ResNet-50 returning its native 2048-D pooled features.

    The backbone uses adaptive global average pooling, so both 224x224 and
    256x256 inputs (and other reasonable spatial sizes) produce the same
    ``[batch, 2048]`` output shape. Inputs are expected in the [0, 1] range;
    ImageNet normalization is applied internally before the backbone.
    """

    def __init__(self, zero_init_residual: bool = True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(
            weights=weights,
            zero_init_residual=zero_init_residual,
        )
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.register_buffer(
            "input_mean",
            torch.tensor(weights.transforms().mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(weights.transforms().std).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_mean) / self.input_std
        return self.backbone(x)


if __name__ == "__main__":
    model = ResNet50().eval()
    with torch.inference_mode():
        for image_size in (224, 256):
            x = torch.rand(2, 3, image_size, image_size)
            out = model(x)
            print(f"Input {image_size}x{image_size} -> output {tuple(out.shape)}")
