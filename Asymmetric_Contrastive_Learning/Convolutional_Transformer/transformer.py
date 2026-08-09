"""
transformer.py — Optimised Transformer Head for Prajna A100 Nodes
-----------------------------------------------------------------
Two key optimisations over the baseline:

1. Flash Attention via F.scaled_dot_product_attention (PyTorch ≥ 2.0)
   ─────────────────────────────────────────────────────────────────────
   The vanilla attention stores the full [B, H, N, N] score matrix in VRAM
   for backprop. For our 1025-token sequence:
     Baseline: [B=32, H=12, N=1025, N=1025] × bf16 = ~750 MB per layer × 6 = 4.5 GB
   
   F.scaled_dot_product_attention dispatches to the FlashAttention-2 CUDA kernel
   on A100 (SM80+). Flash Attention is an IO-aware algorithm that tiles the
   computation and never materialises the full score matrix, reducing attention
   VRAM from O(N²) → O(N). Speed gain: 2–4× faster with lower memory.

2. Gradient Checkpointing
   ─────────────────────────────────────────────────────────────────────
   When use_checkpoint=True, activations are NOT stored during forward —
   they are recomputed during backward instead. Trades ~33% extra compute
   for a large reduction in activation memory (useful if pushing batch size
   beyond 32, or if running other models concurrently on the node).
   Set block.use_checkpoint = True in train.py to enable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CustomMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Flash Attention backend.

    Key change: replaces the manual 5-step SDPA with a single call to
    F.scaled_dot_product_attention(), which the A100 executes via the
    memory-efficient FlashAttention-2 kernel.
    """
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.d_k        = hidden_dim // num_heads   # 64 per head

        # Fused Q, K, V projection: single matmul is more cache-friendly
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim,    bias=True)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [B, N, D]  where N=1025 (1024 patch tokens + 1 CLS), D=768
        """
        B, N, D = Z.shape

        # ── Fused QKV projection ──────────────────────────────────────────────
        qkv = self.qkv_proj(Z)                          # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                 # each [B, N, D]

        # ── Reshape into multi-head form ──────────────────────────────────────
        # [B, N, H, d_k] → [B, H, N, d_k]  (required layout for SDPA)
        q = q.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # ── Flash Attention ───────────────────────────────────────────────────
        # On A100 with PyTorch ≥ 2.0 this dispatches to FlashAttention-2.
        # dropout_p=0.0 during training (contrastive models are sensitive to
        # stochastic attention perturbations; keep it deterministic).
        # scale=None → uses the default 1/sqrt(d_k) scaling.
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask   = None,
            dropout_p   = 0.0,
            is_causal   = False,   # full bidirectional attention
        )
        # context: [B, H, N, d_k]

        # ── Concatenate heads and project ─────────────────────────────────────
        # [B, H, N, d_k] → [B, N, H, d_k] → [B, N, D]
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(context)


class CustomTransformerBlock(nn.Module):
    """
    Complete Transformer layer: Pre-LayerNorm MHSA + Pre-LayerNorm MLP.
    Supports optional gradient checkpointing via self.use_checkpoint.
    """
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = CustomMultiHeadAttention(hidden_dim, num_heads)

        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_dim    = int(hidden_dim * mlp_ratio)   # 3072
        self.mlp   = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

        # Set to True externally to enable gradient checkpointing for this block
        self.use_checkpoint: bool = False

    def _forward_impl(self, Z: torch.Tensor) -> torch.Tensor:
        Z = Z + self.attn(self.norm1(Z))   # residual 1: attention
        Z = Z + self.mlp(self.norm2(Z))    # residual 2: MLP
        return Z

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # use_reentrant=False is the modern (stable) API and avoids
            # issues with in-place operations inside the checkpointed region.
            return checkpoint(self._forward_impl, Z, use_reentrant=False)
        return self._forward_impl(Z)


class SemanticProjector(nn.Module):
    """
    Non-linearly expands the 768-D [CLS] token to the 2048-D Semantic Hash space.

    The LayerNorm after GELU prevents the hash vectors from collapsing onto a
    low-dimensional manifold, which would destabilise the NT-Xent loss.
    """
    def __init__(self, transformer_dim: int = 768, hash_dim: int = 2048):
        super().__init__()
        self.up_project = nn.Sequential(
            nn.Linear(transformer_dim, hash_dim),
            nn.GELU(),
            nn.LayerNorm(hash_dim),
        )

    def forward(self, transformer_out: torch.Tensor) -> torch.Tensor:
        """
        transformer_out: [B, 1025, 768]
        Returns:          [B, 2048]
        """
        cls_token = transformer_out[:, 0, :]   # index 0 is always the [CLS] token
        return self.up_project(cls_token)