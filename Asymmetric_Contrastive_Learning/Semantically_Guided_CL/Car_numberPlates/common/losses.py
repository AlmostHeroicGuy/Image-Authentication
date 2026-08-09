"""Contrastive losses used by both experiments."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Compute the standard SimCLR NT-Xent loss for paired batches."""

    if z1.shape != z2.shape:
        raise ValueError(f"Expected paired tensors with equal shape, got {z1.shape} and {z2.shape}.")
    if z1.ndim != 2:
        raise ValueError("NT-Xent expects [batch, dim] projection tensors.")
    z1 = _gather_with_grad(z1)
    z2 = _gather_with_grad(z2)
    if z1.size(0) < 2:
        raise ValueError("NT-Xent needs at least two positive pairs to provide negatives.")

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    similarity = torch.matmul(z, z.T) / temperature
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(self_mask, float("-inf"))

    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(similarity, targets)


def _gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    gathered = all_gather(tensor)
    return torch.cat(tuple(gathered), dim=0)
