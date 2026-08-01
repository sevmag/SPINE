"""CURTAIN head: query positions cross-attend to the encoded event.

Each held-out query sensor is embedded with Fourier position features and
cross-attends to the backbone tokens (+ CLS) to a per-query embedding. Each
objective then projects that shared embedding to its own outputs (occupancy: a
hit logit; dt: a Delta-t value), so v1 -> v2 adds a head, not head width.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from spine.backbones.base import EncodedEvent
from spine.pretext.base import Objective


class PositionQueryEncoder(nn.Module):
    """NeRF-style Fourier features of a standardized position -> d_model."""

    def __init__(self, d_model: int, n_freq: int = 12):
        """Build the Fourier feature projection.

        Args:
            d_model: Output embedding width.
            n_freq: Number of octave-spaced frequencies per coordinate.
        """
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi)
        self.proj = nn.Sequential(
            nn.Linear(3 + 3 * 2 * n_freq, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pos: Tensor) -> Tensor:
        """Fourier-embed standardized positions.

        Args:
            pos: [B, Q, 3] standardized query positions.

        Returns:
            [B, Q, d_model] position embeddings.
        """
        ang = pos.unsqueeze(-1) * self.freqs  # [B, Q, 3, n_freq]
        feat = torch.cat([pos, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)
        return self.proj(feat)


class QueryCrossAttnEncoder(nn.Module):
    """Shared trunk: queries cross-attend to [CLS; tokens] -> [B, Q, D].

    The per-query embedding each objective projects from; it holds no output
    layer -- objectives own their heads.
    """

    def __init__(self, d_model: int, num_heads: int = 8, mlp_ratio: int = 4):
        """Build the query embedding + cross-attention trunk.

        Args:
            d_model: Embedding width (matches the backbone).
            num_heads: Cross-attention heads.
            mlp_ratio: FFN expansion factor.
        """
        super().__init__()
        self.qpos = PositionQueryEncoder(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, query_pos: Tensor, enc: EncodedEvent) -> Tensor:
        """Cross-attend each query position to the encoded event.

        Args:
            query_pos: [B, Q, 3] standardized query positions.
            enc: The backbone's encoded batch (tokens + mask + CLS).

        Returns:
            [B, Q, D] per-query embeddings.
        """
        kv = torch.cat([enc.cls.unsqueeze(1), enc.tokens], dim=1)  # [B,1+L,D]
        ones = torch.ones(
            enc.token_mask.shape[0], 1, dtype=torch.bool, device=enc.token_mask.device
        )
        kv_real = torch.cat([ones, enc.token_mask], dim=1)  # True=real
        q = self.qpos(query_pos)
        kv_n = self.norm_kv(kv)
        a, _ = self.attn(
            self.norm_q(q), kv_n, kv_n, key_padding_mask=~kv_real, need_weights=False
        )
        q = q + a
        q = q + self.ffn(self.norm2(q))
        return q


class MultiObjectiveHead(nn.Module):
    """Shared query encoder + one head per objective.

    forward returns a list of per-query predictions (one tensor per objective,
    in `objectives` order); each objective built its own head and scores it.
    """

    def __init__(self, d_model: int, objectives: Sequence[Objective]):
        """Build one head per objective over a shared query encoder.

        Args:
            d_model: Embedding width (matches the backbone).
            objectives: The task's objectives, each contributing a head.
        """
        super().__init__()
        self.encoder = QueryCrossAttnEncoder(d_model)
        self.heads = nn.ModuleList([o.build_head(d_model) for o in objectives])

    def forward(self, query_pos: Tensor, enc: EncodedEvent) -> list[Tensor]:
        """Run every objective head off the shared query embedding.

        Args:
            query_pos: [B, Q, 3] standardized query positions.
            enc: The backbone's encoded batch.

        Returns:
            One [B, Q, C_i] prediction tensor per objective, in order.
        """
        emb = self.encoder(query_pos, enc)  # [B, Q, D]
        return [head(emb) for head in self.heads]  # list of [B, Q, C_i]
