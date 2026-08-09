"""
loss.py — Asymmetric NT-Xent Loss with Distributed All-Gather
--------------------------------------------------------------
The key insight for contrastive learning on multi-GPU systems:
  each GPU alone sees BATCH_PER_GPU negatives.
  With all_gather, every GPU sees BATCH_PER_GPU × WORLD_SIZE negatives.
  
  For contrastive learning quality:
    - SimCLR paper showed performance scales strongly with batch size (more negatives = better)
    - 8 × A100 with batch=32 gives 256 effective negatives — a strong training signal.

The gradient trick:
  torch.distributed.all_gather() does NOT propagate gradients to the gathered
  tensors from other ranks. We restore gradient flow only on the local shard by
  replacing gathered[local_rank] with the original tensor before concatenation.

FIX [BUG 4]: The original code summed ALL entries in sim_all (including the
positive pair) into the denominator. In the NT-Xent formulation, the positive
pair must be excluded from the denominator — otherwise the loss penalises the
model for pulling z_orig toward z_global, which is the exact objective we want.

The fix masks out the diagonal (local GPU) or the correct offset block
(distributed) using float('-inf') before taking exp(), so those entries
contribute 0 to the denominator sum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class AsymmetricNTXentLoss(nn.Module):
    """
    Asymmetric Normalized Temperature-Scaled Cross-Entropy Loss.

    Anchors on the pristine image and:
      - pulls it toward its globally-augmented benign view  (positive)
      - pushes it away from the locally-tampered forgery    (hard negative, weighted by alpha)
      - pushes it away from all other images in the batch   (standard negatives)

    In distributed mode, negatives are gathered from ALL GPUs before computing
    the denominator, maximising the contrastive signal without extra communication
    cost during backward (gradient only flows through the local shard).
    """

    def __init__(self, temperature: float = 0.05, alpha: float = 20.0,
                 distributed: bool = False):
        super().__init__()
        self.temperature   = temperature
        self.alpha         = alpha
        self.distributed   = distributed

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensor from every rank and concat along batch dimension.

        Gradient flows only through the local shard (standard trick for
        distributed contrastive losses used in SimCLR, MoCo-v3, etc.).

        Returns: [world_size × local_batch, feature_dim]
        """
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()

        # Allocate receive buffers (no gradient)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)

        # Restore the local shard with gradient so backward still works
        gathered[local_rank] = tensor

        return torch.cat(gathered, dim=0)   # [N_global, D]

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self,
                h_orig:     torch.Tensor,   # [B, 2048]  pristine anchor
                h_global:   torch.Tensor,   # [B, 2048]  benign augmentation (positive)
                h_tampered: torch.Tensor,   # [B, 2048]  localized forgery  (hard negative)
                ) -> torch.Tensor:

        B = h_orig.shape[0]

        # 1. L2-normalise onto the unit hypersphere: dot product = cosine similarity
        z_orig     = F.normalize(h_orig,     dim=1)   # [B, 2048]
        z_global   = F.normalize(h_global,   dim=1)   # [B, 2048]
        z_tampered = F.normalize(h_tampered, dim=1)   # [B, 2048]

        # 2. Gather global views from all GPUs (denominator context)
        if self.distributed and dist.is_initialized() and dist.get_world_size() > 1:
            local_rank   = dist.get_rank()
            world_size   = dist.get_world_size()
            # z_global_all : [N_global, 2048]   — all benign augmentations across cluster
            z_global_all = self._all_gather_with_grad(z_global)
            # The local batch's positives sit at offset [local_rank*B : (local_rank+1)*B]
            pos_start = local_rank * B
        else:
            z_global_all = z_global     # single-GPU fallback
            pos_start    = 0

        N_global = z_global_all.shape[0]

        # ── Numerator (positive pair similarity) ─────────────────────────────
        # Element-wise dot product: [B, 2048] → [B]
        sim_pos = (z_orig * z_global).sum(dim=-1) / self.temperature   # [B]

        

        # ── Hard negative similarity ──────────────────────────────────────────
        sim_hard_neg = (z_orig * z_tampered).sum(dim=-1) / self.temperature   # [B]

        # ── Denominator: anchor vs. ALL global views across every GPU ─────────
        # [B, 2048] × [2048, N_global] → [B, N_global]
        sim_all = torch.matmul(z_orig, z_global_all.T) / self.temperature

        # FIX [BUG 4]: Mask out the positive pair from the denominator.
        # In NT-Xent, the denominator sums over ALL other samples except the
        # positive. Without this mask, exp(sim_pos) is included in the
        # denominator, which directly contradicts the contrastive objective and
        # causes the gradient to partially cancel itself.
        #
        # For each anchor i, its positive lives at column (pos_start + i) in
        # sim_all. We set those entries to -inf so exp(-inf) = 0 — they
        # contribute nothing to the denominator sum.
        pos_indices = torch.arange(B, device=z_orig.device) + pos_start  # [B]
        sim_all[torch.arange(B, device=z_orig.device), pos_indices] = float('-inf')

        # Sum of exp over all negatives (positive is now masked out): [B]
        exp_all_sum = sim_all.exp().sum(dim=1)

        # Hard negative term — weighted by alpha (gravitational repulsor in latent space)
        exp_hard_neg = self.alpha * sim_hard_neg.exp()

        # ── Full denominator ──────────────────────────────────────────────────
        denominator = exp_all_sum + exp_hard_neg   # [B]

        # ── Negative log-likelihood  ──────────────────────────────────────────
        # Computed as (sim_pos - log(denominator)) for numerical stability.
        # Avoids computing exp(sim_pos) / denominator which can overflow.
        loss_per_sample = -(sim_pos - torch.log(denominator))   # [B]

        return loss_per_sample.mean()