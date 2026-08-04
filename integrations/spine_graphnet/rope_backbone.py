"""graphnet DeepIceRope as a SPINE Backbone.

Subclasses DeepIceRope so `state_dict()` IS the transfer artifact -- the
checkpoint's ["backbone"] entry loads straight into a downstream DeepIceRope.
`encode` replays the jagged rotary forward but returns the full padded token
sequence, its mask and the CLS embedding; stock DeepIceRope returns only CLS,
and the CURTAIN head cross-attends each query over the whole pulse sequence.

SPINE pulses are (x, y, z, t, charge); DeepIceRope reads the time coordinate
from column 4 (the NuBench (x, y, z, charge, t) order), so `encode` swaps the
last two columns before the tokenizer and rotation -- otherwise the rotation
would use charge as time.
"""

from __future__ import annotations

import torch
from graphnet.models.transformer.icemix_rope import DeepIceRope
from graphnet.models.utils import array_to_sequence

from spine.backbones.base import Backbone, EncodedEvent


class DeepIceRopeBackbone(DeepIceRope, Backbone):
    """graphnet DeepIceRope exposing SPINE's token-level `encode`."""

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 3,
        head_size: int = 16,
        depth_rel: int = 2,
        seq_length: int = 192,
        mlp_ratio: int = 4,
        rope_per_axis: bool = True,
        scaled_emb: bool = False,
        compile_blocks: bool = False,
    ):
        """Construct the reduced DeepIceRope this repo pretrains.

        Defaults match the v1/v2 CURTAIN DeepIce backbone (d_model=128,
        depth=3, head_size=16, depth_rel=2). Every block is rotary-uniform, so
        DeepIce's `n_rel` (how many blocks carry the relative bias) has no
        analogue here.

        Args:
            d_model: Embedding width (DeepIceRope hidden_dim).
            depth: Number of main rotary blocks.
            head_size: Per-head width; heads = d_model // head_size. Must be
                divisible by 8 (rotation pairs split over 4 coordinates).
            depth_rel: Number of leading rotary blocks (DeepIce's sandwich
                slot); total depth is depth_rel + depth.
            seq_length: Base dimensionality of the Fourier features.
            mlp_ratio: Mlp expansion ratio of the tokenizer and blocks.
            rope_per_axis: Use a separate rotary frequency band per coordinate.
            scaled_emb: Scale the sinusoidal positional embeddings.
            compile_blocks: Wrap the block stack in torch.compile.
        """
        super().__init__(
            hidden_dim=d_model,
            depth=depth,
            head_size=head_size,
            depth_rel=depth_rel,
            seq_length=seq_length,
            mlp_ratio=mlp_ratio,
            n_features=5,
            rope_per_axis=rope_per_axis,
            scaled_emb=scaled_emb,
            compile_blocks=compile_blocks,
        )
        self.out_dim = d_model

    def encode(self, batch: dict) -> EncodedEvent:
        """Run DeepIceRope's jagged rotary forward, exposing tokens + mask + CLS.

        Args:
            batch: Collated batch; `batch["pulses"]` is a jagged NJT [B, *, 5]
                in SPINE (x, y, z, t, charge) order.

        Returns:
            Per-token embeddings [B, L, D], token mask [B, L] (True = real
            pulse) and CLS embedding [B, D], with L the batch's max pulse count.
        """
        pulses = batch["pulses"]
        lengths = pulses.offsets().diff()
        batch_size = lengths.numel()
        # SPINE (x, y, z, t, charge) -> NuBench (x, y, z, charge, t): the rotary
        # code reads time from column 4, so the last two columns must swap.
        feats = pulses.values()[:, [0, 1, 2, 4, 3]]
        batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, device=feats.device), lengths
        )
        x, _, seq_length = array_to_sequence(feats, batch_idx, nested=True)
        x = self.fourier_ext(x, seq_length)
        x = self._prepend_cls_nested(x, batch_idx)
        rope_cos, rope_sin = self._rope_angles(feats, batch_idx, batch_size)
        x = self._blocks_fn(x, rope_cos=rope_cos, rope_sin=rope_sin)
        # to_padded pads to the batch's true max length; the CLS sits at index 0
        # of every event, the per-pulse tokens follow.
        padded = x.to_padded_tensor(0.0)
        token_mask = (
            torch.arange(padded.shape[1] - 1, device=feats.device)[None]
            < lengths[:, None]
        )
        return EncodedEvent(
            tokens=padded[:, 1:], token_mask=token_mask, cls=padded[:, 0]
        )
