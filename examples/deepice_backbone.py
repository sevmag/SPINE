"""graphnet DeepIce as a SPINE Backbone (the graphnet integration example).

Subclasses DeepIce so `state_dict()` IS the transfer artifact -- checkpoint
["backbone"] keys load straight into a downstream stock DeepIce. `encode`
drives the submodules over the padded batch to expose per-token embeddings
(stock DeepIce returns only CLS).

TODO(vendor): the token path reaches into DeepIce internals and is fragile vs
upstream refactors; upstream a return_tokens flag or vendor the encoder.
"""

from __future__ import annotations

import torch
from graphnet.models.gnn import DeepIce

from spine.backbones.base import Backbone, EncodedEvent


class DeepIceBackbone(DeepIce, Backbone):
    """graphnet DeepIce exposing SPINE's token-level `encode`."""

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 3,
        head_size: int = 16,
        depth_rel: int = 2,
        n_rel: int = 2,
        seq_length: int = 192,
    ):
        """Construct the reduced DeepIce this repo pretrains.

        Args:
            d_model: Embedding width (DeepIce hidden_dim).
            depth: Number of CLS-attention blocks.
            head_size: Per-head width; heads = d_model // head_size.
            depth_rel: Number of relative-spacetime blocks.
            n_rel: How many leading blocks receive the relative bias.
            seq_length: Base dimensionality of the Fourier features.
        """
        super().__init__(
            hidden_dim=d_model,
            depth=depth,
            seq_length=seq_length,
            head_size=head_size,
            depth_rel=depth_rel,
            n_rel=n_rel,
            include_dynedge=False,
            n_features=5,
        )
        self.out_dim = d_model

    def encode(self, batch: dict) -> EncodedEvent:
        """Run the DeepIce token-forward over SPINE's jagged batch.

        `to_padded_tensor(0.0)` pads to the batch's true max length by
        construction -- required: FourierEncoder's length embedding expands to
        `max(seq_length)` and must match the token width L.

        Args:
            batch: Collated batch; `batch["pulses"]` is a jagged NJT [B,*,F].

        Returns:
            Per-token embeddings, token mask and CLS embedding.
        """
        pulses = batch["pulses"]
        x0 = pulses.to_padded_tensor(0.0)
        lengths = pulses.offsets().diff()
        mask = torch.arange(x0.shape[1], device=x0.device)[None] < lengths[:, None]
        x = self.fourier_ext(x0, lengths)
        rel_pos_bias = self.rel_pos(x0)
        b = mask.shape[0]
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None
        mask_cls = torch.cat(
            [torch.ones(b, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask_cls.shape, device=mask.device)
        attn_mask[~mask_cls] = -torch.inf
        cls = self.cls_token.weight.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([cls, x], 1)
        for blk in self.blocks:
            x = blk(x, None, attn_mask)
        return EncodedEvent(tokens=x[:, 1:], token_mask=mask, cls=x[:, 0])
