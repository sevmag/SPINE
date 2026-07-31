"""SSLModule: backbone + pretext head, task delegates the loss.

Task-agnostic. It wires a `Backbone` to a `PretextTask`'s head, calls
`task.loss` in train/val, and owns optimizer/scheduler plus the DDP-correctness
bit: `sync_dist=True` on the val metric so `ReduceLROnPlateau` steps identically
on every rank (per-rank val desyncs replica LRs and silently corrupts DDP).

`self.backbone` / `self.model` are exposed for `TransferCheckpoint` (backbone =
the exported encoder; model = backbone+head for the full checkpoint). They are
properties over one `ModuleDict` so params aren't registered twice.
"""

from __future__ import annotations

import pytorch_lightning as pl
from torch import nn

from spine.backbones.base import Backbone
from spine.engine.optim import build_optimizer
from spine.pretext.base import PretextTask


class SSLModule(pl.LightningModule):
    def __init__(self, backbone: Backbone, task: PretextTask,
                 lr: float = 5e-4, lr_patience: int = 7):
        super().__init__()
        self.task = task
        self.model = nn.ModuleDict(
            {"backbone": backbone, "head": task.build_head(backbone.out_dim)}
        )
        self.lr = lr
        self.lr_patience = lr_patience

    @property
    def backbone(self) -> nn.Module:
        return self.model["backbone"]

    @property
    def head(self) -> nn.Module:
        return self.model["head"]

    def forward(self, batch):
        enc = self.backbone.encode(batch)
        return self.head(batch["qpos"].to_padded_tensor(0.0), enc)

    def _step(self, batch, stage: str):
        if batch is None:  # whole batch was unusable events
            return None
        loss, metrics = self.task.loss(self(batch), batch)
        bs = int(batch["label"].values().numel())
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"),
                 on_epoch=True, sync_dist=True, batch_size=bs)
        for k, v in metrics.items():
            self.log(f"{stage}_{k}", v, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        return build_optimizer(self.parameters(), self.lr, self.lr_patience)
