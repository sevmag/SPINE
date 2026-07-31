"""CURTAIN head: query positions cross-attend to the encoded event.

Each held-out query sensor is embedded with Fourier position features and
cross-attends to the backbone tokens (+ CLS) to emit `channels` outputs per
query: 1 for occupancy (v1), 2 for occupancy + Delta-t (v2). The channel count
comes from the task's objectives, so v1 -> v2 is a width change, not a new head.
Ported from the occupancy study's cross-attention head.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from spine.backbones.base import EncodedEvent


class PositionQueryEncoder(nn.Module):
    """NeRF-style Fourier features of a standardized position -> d_model."""

    def __init__(self, d_model: int, n_freq: int = 12):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi)
        self.proj = nn.Sequential(
            nn.Linear(3 + 3 * 2 * n_freq, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pos: Tensor) -> Tensor:
        ang = pos.unsqueeze(-1) * self.freqs  # [B, Q, 3, n_freq]
        feat = torch.cat([pos, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)
        return self.proj(feat)


class QueryCrossAttnHead(nn.Module):
    """Queries cross-attend to [CLS; tokens] -> per-query `channels` outputs."""

    def __init__(self, d_model: int, channels: int, num_heads: int = 8,
                 mlp_ratio: int = 4):
        super().__init__()
        self.qpos = PositionQueryEncoder(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )
        self.out = nn.Linear(d_model, channels)

    def forward(self, query_pos: Tensor, enc: EncodedEvent) -> Tensor:
        """`query_pos` [B,Q,3] standardized -> [B,Q,channels]."""
        kv = torch.cat([enc.cls.unsqueeze(1), enc.tokens], dim=1)   # [B,1+L,D]
        ones = torch.ones(enc.token_mask.shape[0], 1, dtype=torch.bool,
                          device=enc.token_mask.device)
        kv_real = torch.cat([ones, enc.token_mask], dim=1)          # True=real
        q = self.qpos(query_pos)
        kv_n = self.norm_kv(kv)
        a, _ = self.attn(self.norm_q(q), kv_n, kv_n,
                         key_padding_mask=~kv_real, need_weights=False)
        q = q + a
        q = q + self.ffn(self.norm2(q))
        return self.out(q)
