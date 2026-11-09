"""DFlash block drafter: a parallel K-token speculator.

Ported from ``spec_decode/dflash.py`` (single-script original), which follows
z-lab/dflash — Diffusion-inspired Flash Speculative Decoding.

- target feature down-projection and fusion
- parallel K-token block prediction in one forward pass
- trainable slot positional queries
- causal context + bidirectional block attention mask
- emits ``[N, block_size, V]`` logits, i.e. exactly what ``losses/`` consumes

The training objective is a hook (``compute_loss(..., loss_fn=...)``) so the same
drafter can be trained with plain CE, with this folder's fused DFlash objective
(:mod:`models.dflash.loss`) or with D-PACE (:mod:`losses.dpace`).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFlashAttention(nn.Module):
    """Multi-head attention supporting causal context prefix and bidirectional
    attention within the speculative block tokens."""

    def __init__(self, d_model: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=qkv_bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, S, d_model); attn_mask: (B, 1, S, S) or (1, 1, S, S) additive mask."""
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)


class DFlashTransformerLayer(nn.Module):
    """Pre-LN transformer block for DFlash."""

    def __init__(self, d_model: int, num_heads: int, d_ffn: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = DFlashAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn, bias=False),
            nn.SiLU(),
            nn.Linear(d_ffn, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class DFlashSpeculator(nn.Module):
    """Original DFlash non-autoregressive drafter.

    - reuses the target model's intermediate hidden representations
    - projects target features down to a smaller drafter dimension
    - appends K trainable block queries to predict K future tokens in parallel
    - attention mask: causal over context, block tokens see all context and all
      block tokens (bidirectional)
    - emits K draft logits in a single feedforward pass
    """

    def __init__(
        self,
        d_target: int,
        d_model: int,
        target_vocab_size: int,
        block_size: int = 4,
        num_layers: int = 3,
        num_heads: int = 4,
        target_lm_head: Optional[nn.Linear] = None,
        norm_before_head: bool = True,
    ):
        super().__init__()
        self.d_target = d_target
        self.d_model = d_model
        self.block_size = block_size
        self.target_vocab_size = target_vocab_size

        self.proj_in = nn.Linear(d_target, d_model, bias=False)

        self.slot_queries = nn.Parameter(torch.randn(1, block_size, d_model) * 0.02)

        self.layers = nn.ModuleList([
            DFlashTransformerLayer(d_model=d_model, num_heads=num_heads, d_ffn=d_model * 4)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.RMSNorm(d_model)

        self.proj_out = nn.Linear(d_model, d_target, bias=False)
        self.final_norm = nn.RMSNorm(d_target) if norm_before_head else None

        if target_lm_head is not None:
            self.lm_head = target_lm_head
            self.lm_head.weight.requires_grad = False   # keep the target head frozen
        else:
            self.lm_head = nn.Linear(d_target, target_vocab_size, bias=False)

    def _build_dflash_mask(self, S_ctx: int, S_block: int, device: torch.device) -> torch.Tensor:
        """Context tokens are causal; block tokens see all context and all block tokens."""
        total_len = S_ctx + S_block
        mask = torch.full((total_len, total_len), float("-inf"), device=device)
        mask[:S_ctx, :S_ctx] = torch.triu(
            torch.full((S_ctx, S_ctx), float("-inf"), device=device), diagonal=1
        )
        mask[S_ctx:, :] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        """target_features: (B, S_ctx, d_target) -> draft_logits: (B, block_size, V)."""
        B, S_ctx, _ = target_features.shape

        h_ctx = self.proj_in(target_features)                 # (B, S_ctx, d_model)
        h_block = self.slot_queries.expand(B, -1, -1)         # (B, block_size, d_model)
        h = torch.cat([h_ctx, h_block], dim=1)                # (B, S_ctx+block_size, d_model)

        attn_mask = self._build_dflash_mask(S_ctx, self.block_size, target_features.device)
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)
        h = self.norm_out(h)

        block_rep = h[:, S_ctx:, :]                           # (B, block_size, d_model)
        block_feat = self.proj_out(block_rep)                 # (B, block_size, d_target)
        if self.final_norm is not None:
            block_feat = self.final_norm(block_feat)
        return self.lm_head(block_feat)                       # (B, block_size, V)

    def compute_loss(
        self,
        target_features: torch.Tensor,
        ground_truth_block_tokens: torch.Tensor,
        loss_fn=None,
    ) -> torch.Tensor:
        """Training loss over the K parallel slots.

        loss_fn(logits, targets) -> scalar lets the objective be swapped:
        ``models.dflash.loss.dflash_loss``, ``losses.dpace.dpace_loss``, or the
        plain cross entropy used when it is None.
        """
        logits = self.forward(target_features)                # (B, block_size, V)
        if loss_fn is not None:
            return loss_fn(logits, ground_truth_block_tokens)
        return F.cross_entropy(
            logits.view(-1, self.target_vocab_size),
            ground_truth_block_tokens.view(-1),
        )

    @torch.no_grad()
    def draft(self, target_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-pass speculative drafting: (draft_tokens, draft_logits)."""
        logits = self.forward(target_features)
        return torch.argmax(logits, dim=-1), logits
