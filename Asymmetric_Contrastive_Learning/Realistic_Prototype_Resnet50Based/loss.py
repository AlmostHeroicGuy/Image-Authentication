"""Asymmetric contrastive objective for pristine, benign, and forged views."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricNTXentLoss(nn.Module):
    """
    Contrast a pristine anchor against benign and locally forged views.

    For every pristine embedding, the matching benign embedding is the
    positive. Other benign embeddings are ordinary negatives, while the
    matching forged embedding is an explicitly weighted hard negative.

    For normalized embeddings, the per-sample objective is::

        -s_positive + log(
            sum(exp(s_other_benign)) + alpha * exp(s_forged)
        )

    where each similarity ``s`` is cosine similarity divided by temperature.
    The matching positive is intentionally excluded from the denominator in
    this asymmetric variant.

    In distributed training, benign embeddings from every rank contribute
    ordinary negatives. Gradients are retained for the local rank's embeddings;
    embeddings received from other ranks are treated as fixed context.
    """

    def __init__(
        self,
        temperature: float = 0.05,
        alpha: float = 20.0,
        distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distributed = distributed

    @staticmethod
    def _all_gather_with_local_grad(tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather equal-sized batches from all ranks along the batch dimension.

        ``dist.all_gather`` does not preserve autograd connections. Replacing
        the local gathered copy with the original tensor restores gradients for
        the local shard while keeping remote shards as detached negatives.
        """
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()

        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        gathered[local_rank] = tensor

        return torch.cat(gathered, dim=0)

    def forward(
        self,
        h_orig: torch.Tensor,
        h_global: torch.Tensor,
        h_tampered: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the scalar batch loss from three ``[batch, feature]`` tensors.

        ``h_orig`` contains pristine anchors, ``h_global`` their matching
        benign views, and ``h_tampered`` their matching local forgeries.
        """
        batch_size = h_orig.shape[0]

        # Normalization makes every dot product a cosine similarity.
        z_orig = F.normalize(h_orig, dim=1)
        z_global = F.normalize(h_global, dim=1)
        z_tampered = F.normalize(h_tampered, dim=1)

        # Build the ordinary-negative pool and locate each local positive in it.
        use_distributed_pool = (
            self.distributed
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if use_distributed_pool:
            local_rank = dist.get_rank()
            z_global_all = self._all_gather_with_local_grad(z_global)
            positive_start = local_rank * batch_size
        else:
            z_global_all = z_global
            positive_start = 0

        positive_similarity = (
            (z_orig * z_global).sum(dim=-1) / self.temperature
        )
        forged_similarity = (
            (z_orig * z_tampered).sum(dim=-1) / self.temperature
        )
        benign_similarities = (
            torch.matmul(z_orig, z_global_all.T) / self.temperature
        )

        # Remove each anchor's matching benign view from the negative pool.
        row_indices = torch.arange(batch_size, device=z_orig.device)
        positive_columns = row_indices + positive_start
        benign_similarities[row_indices, positive_columns] = float("-inf")

        ordinary_negative_sum = benign_similarities.exp().sum(dim=1)
        hard_negative_term = self.alpha * forged_similarity.exp()
        denominator = ordinary_negative_sum + hard_negative_term

        loss_per_sample = -positive_similarity + torch.log(denominator)
        return loss_per_sample.mean()
