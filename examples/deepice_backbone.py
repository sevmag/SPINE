"""Reference backbone: graphnet's DeepIce behind SPINE's `Backbone` interface.

The graphnet integration example -- it lives in examples/, NOT in the agnostic
core (`src/spine/` imports no graphnet). It implements `Backbone.encode`, so
a launcher instantiates it (Hydra `_target_`) like any custom backbone. The exported
checkpoint is a plain DeepIce state_dict, so the finetuning bench loads it
straight into graphnet DeepIce.

graphnet's `DeepIce.forward` returns only CLS and consumes a torch_geometric
`Data`; the pretext needs the full token sequence and collates a jagged nested
tensor, so `encode` pads it (`to_padded_tensor`) and drives the encoder -- no
`Data`, no `array_to_sequence` -- and returns the token sequence in an
`EncodedEvent`. graphnet is imported lazily -- importing this module to register
the name does not need it; constructing the backbone does.

TODO(vendor): the token path reaches into DeepIce internals
(fourier_ext/rel_pos/sandwich/blocks) and is fragile vs upstream refactors.
Upstream a `return_tokens=True`, or vendor a standalone encoder behind this same
`Backbone` interface -- nothing else would change.
"""

from __future__ import annotations

import torch

from spine.backbones.base import Backbone, EncodedEvent


class DeepIceBackbone(Backbone):
    def __init__(self, d_model: int = 128, depth: int = 3, head_size: int = 16,
                 depth_rel: int = 2, n_rel: int = 2, seq_length: int = 192):
        super().__init__()
        try:
            from graphnet.models.gnn import DeepIce
        except ImportError as e:  # graphnet is env-provided, not a pip dep
            raise ImportError(
                "the 'deepice' backbone requires graphnet on PYTHONPATH "
                "(provided by the graphnet_torch26 env)."
            ) from e
        self.out_dim = d_model
        self.n_rel = n_rel
        self._enc = DeepIce(
            hidden_dim=d_model, depth=depth, seq_length=seq_length,
            head_size=head_size, depth_rel=depth_rel, n_rel=n_rel,
            include_dynedge=False, n_features=5,
        )

    def encode(self, batch) -> EncodedEvent:
        """Ported DeepIce token-forward over SPINE's jagged batch.

        `batch["pulses"]` is a jagged nested tensor [B, *, F];
        `to_padded_tensor(0.0)` gives the dense [B, L, F] DeepIce wants, padded
        to the batch's true max event length by construction -- exactly what
        FourierEncoder needs, since it concatenates a length embedding expanded
        to `max(seq_length)` with per-token embeddings of width L (they line up
        only when `max(seq_length) == L`). `seq_length` and the token mask come
        from the NJT's offsets.
        """
        enc = self._enc
        pulses = batch["pulses"]
        x0 = pulses.to_padded_tensor(0.0)
        lengths = pulses.offsets().diff()
        mask = (torch.arange(x0.shape[1], device=x0.device)[None]
                < lengths[:, None])
        seq_length = lengths
        x = enc.fourier_ext(x0, seq_length)
        rel_pos_bias = enc.rel_pos(x0)
        b = mask.shape[0]
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        for i, blk in enumerate(enc.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None
        mask_cls = torch.cat(
            [torch.ones(b, 1, dtype=mask.dtype, device=mask.device), mask], 1)
        attn_mask = torch.zeros(mask_cls.shape, device=mask.device)
        attn_mask[~mask_cls] = -torch.inf
        cls = enc.cls_token.weight.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([cls, x], 1)
        for blk in enc.blocks:
            x = blk(x, None, attn_mask)
        return EncodedEvent(tokens=x[:, 1:], token_mask=mask, cls=x[:, 0])
