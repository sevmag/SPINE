"""Per-hit Neptune backbone: the masked-point-modeling encoder of Yu, Kamp &
Arguelles, "Reducing Simulation Dependence in Neutrino Telescopes with Masked
Point Transformers" (arXiv:2510.01733).

This mirrors that paper's reference implementation -- the ``prometheus`` branch
of github.com/felixyu7/neptune -- not the latest upstream. The current ``main``
branch has since moved to a different design (parameter-free 4D RoPE and FPS
point-cloud patchification), so this file is pinned to the published
architecture to keep comparisons faithful to the paper's method rather than the
evolving one.

Each hit is its own token: the paper's below-``max_tokens`` path, so there is no
FPS patchification here. Content and position are kept separate on purpose --
the per-hit MLP embeds only non-positional features (charge), while a learned 4D
MLP embeds absolute space-time position and is ADDED to the token. A plain
Transformer encoder mixes tokens and a masked mean over real hits forms the
event embedding (the paper has no CLS token). Holding position out of the token
content is what makes a masked-position pretext non-trivial: with charge alone
visible, the encoder must infer a hit's location -- see ``spine.pretrain.mpm``.

``encode`` honors an optional ``batch["pos_mask"]`` (bool ``[B, L]`` over pulses)
with ``batch["pos_mask_mode"]`` ("spatial" | "temporal" | "spatiotemporal"),
swapping masked hits' coordinates for learned mask embeddings before the
positional MLP; CURTAIN and fine-tuning leave it unset and run unmasked.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from spine.backbones.base import Backbone, EncodedEvent


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """GELU/LayerNorm MLP stack ending in a linear projection to ``out_dim``."""
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.GELU(), nn.LayerNorm(h)]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class NeptuneBackbone(Backbone):
    """Per-hit Neptune encoder (paper / ``prometheus`` architecture).

    ``pos_cols`` index the space-time coordinates in the pulse feature vector as
    ``(x, y, z, t)`` (spatial first, time last); ``content_cols`` index the
    position-free features the token MLP sees (charge).
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 3,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        pos_cols: tuple[int, ...] = (0, 1, 2, 3),
        content_cols: tuple[int, ...] = (4,),
        pos_hidden: tuple[int, ...] = (64, 256),
        content_hidden: tuple[int, ...] = (256,),
    ):
        """Build the per-hit content/positional MLPs and the transformer.

        Args:
            d_model: Token / embedding width.
            depth: Number of transformer encoder layers.
            n_heads: Attention heads.
            dim_feedforward: Encoder feed-forward width.
            dropout: Encoder dropout.
            pos_cols: Feature columns for space-time position, spatial then time.
            content_cols: Feature columns for the position-free token content.
            pos_hidden: Hidden widths of the positional MLP.
            content_hidden: Hidden widths of the content MLP.
        """
        super().__init__()
        self.out_dim = d_model
        self.pos_cols = list(pos_cols)
        self.content_cols = list(content_cols)
        self.content_mlp = _mlp(len(content_cols), content_hidden, d_model)
        self.pos_mlp = _mlp(len(pos_cols), pos_hidden, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.ln = nn.LayerNorm(d_model)
        # "Position unknown" tokens in raw coordinate space, swapped in for
        # masked hits before the positional MLP (masked point modeling).
        self.spatial_mask_emb = nn.Parameter(torch.randn(3) * 0.02)
        self.time_mask_emb = nn.Parameter(torch.randn(1) * 0.02)

    def encode(self, batch: dict) -> EncodedEvent:
        """Encode a collated batch of jagged pulses into per-hit tokens."""
        pulses = batch["pulses"]
        x0 = pulses.to_padded_tensor(0.0)
        lengths = pulses.offsets().diff()
        length = x0.shape[1]
        mask = torch.arange(length, device=x0.device)[None] < lengths[:, None]
        pos = x0[..., self.pos_cols]
        content = x0[..., self.content_cols]

        pos_mask = batch.get("pos_mask")
        if pos_mask is not None:
            mode = batch.get("pos_mask_mode", "spatiotemporal")
            m = (pos_mask & mask).unsqueeze(-1)
            xyz, t = pos[..., 0:3], pos[..., 3:4]
            if mode in ("spatial", "spatiotemporal"):
                xyz = torch.where(m, self.spatial_mask_emb, xyz)
            if mode in ("temporal", "spatiotemporal"):
                t = torch.where(m, self.time_mask_emb, t)
            pos = torch.cat([xyz, t], dim=-1)

        tok = self.content_mlp(content) + self.pos_mlp(pos)
        tok = self.encoder(tok, src_key_padding_mask=~mask)
        tok = self.ln(tok)
        denom = mask.sum(1, keepdim=True).clamp(min=1)
        cls = (tok * mask.unsqueeze(-1)).sum(1) / denom
        return EncodedEvent(tokens=tok, token_mask=mask, cls=cls)
