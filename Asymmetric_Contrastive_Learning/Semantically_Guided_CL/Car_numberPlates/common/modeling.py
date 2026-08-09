"""ResNet-50 encoder and standard SimCLR projection head."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ProjectionHead(nn.Module):
    """The standard two-layer SimCLR projection head."""

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SimCLRModel(nn.Module):
    """ResNet-50 encoder with a shared SimCLR projection head."""

    def __init__(
        self,
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection_head = ProjectionHead(self.feature_dim, hidden_dim, projection_dim)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.forward_features(images))

    @torch.no_grad()
    def extract_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """Return the ResNet feature before the projection head."""

        was_training = self.training
        self.eval()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        features = self.forward_features(image)
        if was_training:
            self.train()
        return features
