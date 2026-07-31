"""Backbone interface: a batch of pulse sets -> per-token embeddings.

The backbone is the swappable transformer encoder. It knows nothing about the
pretext task: it turns a batch of variable-length pulse sets into a padded
sequence of token embeddings + a mask, plus a pooled CLS embedding. Pretext
heads consume `EncodedEvent`, so swapping DeepIce for another architecture is
implementing this one method -- nothing in `pretext/` or `engine/` changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class EncodedEvent:
    """What every backbone returns."""

    tokens: Tensor      # [B, L, D] per-pulse token embeddings
    token_mask: Tensor  # [B, L] bool, True = real pulse (not padding)
    cls: Tensor         # [B, D] pooled event embedding


class Backbone(nn.Module, ABC):
    """Encoder contract. Subclasses set `out_dim` and implement `encode`."""

    out_dim: int

    @abstractmethod
    def encode(self, batch: dict) -> EncodedEvent:
        """Encode one collated batch into an `EncodedEvent`.

        `batch` is the pretext task's collate output. Every backbone may rely on
        two keys: `batch["x"]` [B, L, F] zero-padded pulse features and
        `batch["token_mask"]` [B, L] bool (True = real pulse). Tasks add their
        own keys (query positions, labels, ...) that the head/loss consume; the
        backbone touches only these two. Plain tensors -- no graph-library type
        -- so the core stays reader-agnostic.
        """
        raise NotImplementedError
