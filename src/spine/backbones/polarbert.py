"""PolarBERT backbone: the masked-DOM-prediction encoder of Timiryasov, Tastet
& Ruchayskiy, "PolarBERT: A Foundation Model for IceCube" (NeurIPS ML4PS 2024).

This mirrors the paper's model as implemented in github.com/timinar/PolarBERT
with every post-paper option (RoPE, QK-norm, muP, RMSNorm, DOM-position
embeddings) at its off/paper default, so comparisons stay faithful to the
published method rather than the evolving repository.

Each pulse is one token, ordered by time. The token concatenates two halves: a
learned per-sensor id embedding (the "natural tokenization" the paper builds
on -- a vocabulary over the detector's optical modules, with index 0 reserved
for padding and a dedicated learned mask vector) and a linear embedding of the
remaining pulse features. A learned CLS ("regression") token is prepended and
carries the event-level readout; there are no positional encodings -- the
timestamp feature orders the sequence. The trunk is a pre-LN encoder with
bias-free q/k/v/out projections and a two-layer MLP, closed by a final
LayerNorm.

Detector adaptations, by construction of the NuBench hexagon data: the
auxiliary flag does not exist here (every pulse is "primary", so the paper's
primaries-only masking and auxiliary-aware downsampling are vacuous), and the
feature half embeds (time, charge) in this project's conventions --
charge-weighted-mean-referenced time in microseconds and log10(1+q) charge --
instead of the reference's Kaggle-window constants.

``encode`` honors an optional ``batch["dom_mask"]`` (bool ``[B, L]`` over
pulses), swapping masked pulses' sensor embedding for the learned mask vector
(masked DOM prediction); CURTAIN-style finetuning leaves it unset.

Constructor defaults are the paper's efficient 8M configuration (d_model 256,
8 layers, 8 heads, MLP hidden 1024, sensor-embedding half d_model/2);
configs/backbone/polarbert_match.yaml scales the trunk to the reduced-DeepIce
mirror for capacity-matched comparisons.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spine.backbones.base import Backbone, EncodedEvent

_INIT_STD = 0.02  # the reference initializes its special embeddings at N(0, 0.02)


def _feature_embed(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Module:
    """Linear feature embedding; hidden widths add ReLU layers before it.

    The paper embeds the non-sensor features with a single linear map; hidden
    widths exist only for capacity matching (the embedding side is the one
    place parameter count may flex when mirroring another encoder's budget).
    """
    if not hidden:
        return nn.Linear(in_dim, out_dim)
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class _Block(nn.Module):
    """Pre-LN encoder block with bias-free attention projections."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, activation: str):
        super().__init__()
        self.n_heads = n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), act, nn.Linear(dim_feedforward, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        h = self.ln1(x)
        b, s, d = h.shape
        hd = d // self.n_heads
        q = self.wq(h).view(b, s, self.n_heads, hd).transpose(1, 2)
        k = self.wk(h).view(b, s, self.n_heads, hd).transpose(1, 2)
        v = self.wv(h).view(b, s, self.n_heads, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(b, s, d)
        x = x + self.wo(o)
        return x + self.ff(self.ln2(x))


class PolarBERTBackbone(Backbone):
    """Per-pulse PolarBERT encoder (paper architecture).

    Tokens = [sensor-id embedding | feature embedding]; a learned CLS token is
    prepended and read out as the event embedding.
    """

    def __init__(
        self,
        d_model: int = 256,
        depth: int = 8,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        num_sensors: int = 5160,
        dom_embed_dim: int | None = None,
        feat_hidden: tuple[int, ...] = (),
        n_features: int = 2,
        activation: str = "relu",
    ):
        """Build the sensor/feature embeddings and the pre-LN trunk.

        Args:
            d_model: Token width.
            depth: Encoder layers.
            n_heads: Attention heads.
            dim_feedforward: MLP hidden width in each block.
            num_sensors: Sensor-id vocabulary size (hexagon: 5160). Index 0 of
                the embedding table is the padding vector; real ids shift +1.
            dom_embed_dim: Width of the sensor-embedding half; None = the
                paper's d_model/2.
            feat_hidden: Hidden widths of the feature embedding (empty = the
                paper's single linear map; used for capacity matching only).
            n_features: Non-sensor features per pulse (time, charge).
            activation: Block MLP activation, "relu" (paper) or "gelu".
        """
        super().__init__()
        dom_embed_dim = d_model // 2 if dom_embed_dim is None else dom_embed_dim
        self.out_dim = d_model
        self.num_sensors = num_sensors
        self.dom_embedding = nn.Embedding(num_sensors + 1, dom_embed_dim, padding_idx=0)
        self.mask_token = nn.Parameter(torch.empty(dom_embed_dim).normal_(0.0, _INIT_STD))
        self.cls_embedding = nn.Parameter(torch.empty(1, 1, d_model).normal_(0.0, _INIT_STD))
        self.feature_embedding = _feature_embed(n_features, tuple(feat_hidden), d_model - dom_embed_dim)
        self.blocks = nn.ModuleList(
            _Block(d_model, n_heads, dim_feedforward, activation) for _ in range(depth)
        )
        self.ln = nn.LayerNorm(d_model)

    def encode(self, batch: dict) -> EncodedEvent:
        """Encode a collated batch of jagged pulses into per-pulse tokens + CLS."""
        feats = batch["features"]
        dom_ids = batch["dom_ids"]
        lengths = feats.offsets().diff()
        # An NJT without cached max_seqlen pads the jagged dim to offsets[-1]
        # (the whole batch's pulse count), so the padded views are sliced to
        # the true longest event.
        length = int(lengths.max())
        x0 = feats.to_padded_tensor(0.0)[:, :length]
        ids = dom_ids.to_padded_tensor(0)[:, :length]
        mask = torch.arange(length, device=x0.device)[None] < lengths[:, None]

        dom = self.dom_embedding(torch.where(mask, ids + 1, torch.zeros_like(ids)))
        dom_mask = batch.get("dom_mask")
        if dom_mask is not None:
            m = dom_mask[:, :length].to(mask.device) & mask
            dom = torch.where(m.unsqueeze(-1), self.mask_token.to(dom.dtype), dom)
        tok = torch.cat([dom, self.feature_embedding(x0)], dim=-1)

        b = tok.shape[0]
        tok = torch.cat([self.cls_embedding.expand(b, -1, -1).to(tok.dtype), tok], dim=1)
        keep = torch.cat([mask.new_ones(b, 1), mask], dim=1)
        attn_mask = keep[:, None, None, :]  # True = attend; CLS sees everything
        for blk in self.blocks:
            tok = blk(tok, attn_mask)
        tok = self.ln(tok)
        return EncodedEvent(tokens=tok[:, 1:], token_mask=mask, cls=tok[:, 0])
