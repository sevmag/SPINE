"""Backbone interface: a batch of pulse sets -> per-token embeddings.

The backbone is the swappable transformer encoder. It knows nothing about the
pretext task: it turns a batch of variable-length pulse sets into a padded
sequence of token embeddings + a mask, plus a pooled CLS embedding. Pretext
heads consume `EncodedEvent`, so swapping DeepIce for another architecture is
implementing this one method -- nothing in `pretext/` or `engine/` changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class EncodedEvent:
    """What every backbone returns."""

    tokens: Tensor  # [B, L, D] per-pulse token embeddings
    token_mask: Tensor  # [B, L] bool, True = real pulse (not padding)
    cls: Tensor  # [B, D] pooled event embedding


class Backbone(nn.Module):
    """Encoder contract. Subclasses set `out_dim` and implement `encode`.

    Transfer invariant: the checkpoint's ["backbone"] entry is this module's
    own state_dict, and downstream finetuning loads it directly into its
    encoder -- so EXTEND the encoder (subclass it), don't wrap it. Wrapping
    would nest the encoder one attribute deep and prefix every checkpoint key,
    which the downstream load cannot match.
    """

    out_dim: int

    def encode(self, batch: dict) -> EncodedEvent:
        """Encode one collated batch into an `EncodedEvent`.

        `batch` is the pretext task's collate output. Every backbone may rely on
        `batch["pulses"]`, a jagged nested tensor [B, *, F] of per-event pulse
        features with no padding baked in: a padding backbone calls
        `to_padded_tensor(0.0)` and reads `offsets()` for lengths/mask, a varlen
        backbone reads `offsets()` directly. Tasks add their own keys (query
        positions, labels, ...) for the head/loss; the backbone touches only
        `pulses`. A nested tensor -- no graph-library type -- so the core stays
        reader-agnostic.
        """
        raise NotImplementedError
