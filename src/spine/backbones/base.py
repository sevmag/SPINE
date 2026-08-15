"""Backbone interface: a collated batch -> per-token embeddings + CLS.

Swapping encoders means implementing `encode`; pretrain and engine code stay
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class EncodedEvent:
    """What every backbone returns.

    ``cls`` is optional: a backbone with no event-level readout leaves it
    ``None``, and consumers that need an event embedding (the query encoder)
    must guard for it. Per-token pretexts read ``tokens`` and ignore ``cls``.
    """

    tokens: Tensor  # [B, L, D] per-pulse token embeddings
    token_mask: Tensor  # [B, L] bool, True = real pulse (not padding)
    cls: Tensor | None = None  # [B, D] pooled event embedding, or None


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

        Args:
            batch: The task's collate output; backbones touch only
                `batch["pulses"]`, a jagged NJT [B, *, F] with no padding
                baked in (`to_padded_tensor(0.0)` for dense encoders,
                `offsets()` for varlen ones).

        Returns:
            Per-token embeddings, token mask and pooled event embedding.

        Raises:
            NotImplementedError: Subclasses must implement the encoding.
        """
        raise NotImplementedError
