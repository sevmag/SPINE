"""DeepIce backbone (graphnet) exposing per-token embeddings + mask + CLS.

graphnet's `DeepIce.forward` returns only CLS; the pretext needs the full token
sequence, so `encode` re-runs the encoder and returns an `EncodedEvent`. graphnet
is imported lazily (constructing the backbone needs it; importing `spine` does
not). It consumes a torch_geometric `Data(x=[SumP,5], batch=[SumP])` -- the
datamodule builds that (or we drop tg for a padded-tensor path; see DESIGN).

TODO(vendor): the token path reaches into DeepIce internals
(fourier_ext/rel_pos/sandwich/blocks) and is fragile vs upstream refactors.
Upstream a `return_tokens=True`, or vendor a standalone encoder behind this
same `Backbone` interface -- nothing else in the repo would change.
"""

from __future__ import annotations

import torch

from spine.backbones.base import Backbone, EncodedEvent
from spine.backbones.registry import BACKBONES


@BACKBONES.register("deepice")
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
        """`batch` is a torch_geometric Data(x, batch). Ported token forward."""
        from graphnet.models.utils import array_to_sequence

        enc = self._enc
        x0, mask, seq_length = array_to_sequence(batch.x, batch.batch,
                                                 padding_value=0)
        x = enc.fourier_ext(x0, seq_length)
        rel_pos_bias = enc.rel_pos(x0)
        b = mask.shape[0]
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        for i, blk in enumerate(enc.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None
        token_mask = mask  # [B, L] True = real pulse
        mask_cls = torch.cat(
            [torch.ones(b, 1, dtype=mask.dtype, device=mask.device), mask], 1)
        attn_mask = torch.zeros(mask_cls.shape, device=mask.device)
        attn_mask[~mask_cls] = -torch.inf
        cls = enc.cls_token.weight.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([cls, x], 1)
        for blk in enc.blocks:
            x = blk(x, None, attn_mask)
        return EncodedEvent(tokens=x[:, 1:], token_mask=token_mask, cls=x[:, 0])
