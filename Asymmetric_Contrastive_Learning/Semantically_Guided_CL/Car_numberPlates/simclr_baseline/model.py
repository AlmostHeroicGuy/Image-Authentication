"""Baseline model definition."""

import torch

from common.modeling import ProjectionHead, SimCLRModel


def extract_embedding(model: SimCLRModel, image: torch.Tensor) -> torch.Tensor:
    """Return the ResNet feature before the projection head."""

    return model.extract_embedding(image)


__all__ = ["ProjectionHead", "SimCLRModel", "extract_embedding"]
