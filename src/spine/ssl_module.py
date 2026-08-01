"""SSLModule: backbone + pretext head, task delegates the loss.

Task-agnostic. It wires a `Backbone` to a `PretextTask`'s head, calls
`task.loss` in train/val, and owns the optimizer/scheduler plus the
DDP-correctness bit: `sync_dist=True` on the val metric so `ReduceLROnPlateau`
steps identically on every rank (per-rank val desyncs replica LRs and silently
corrupts DDP).

`self.backbone` / `self.model` are exposed for `TransferCheckpoint`
(`spine.utils`): backbone = the exported encoder, model = backbone+head for the
full checkpoint. They are properties over one `ModuleDict` so params aren't
registered twice.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn

from spine.backbones.base import Backbone
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
        self._val_cache = []

    @property
    def backbone(self) -> nn.Module:
        return self.model["backbone"]

    @property
    def head(self) -> nn.Module:
        return self.model["head"]

    def forward(self, batch):
        enc = self.backbone.encode(batch)
        return self.head(batch["qpos"].to_padded_tensor(0.0), enc)

    def training_step(self, batch, _):
        loss, metrics = self.task.loss(self(batch), batch)
        bs = int(batch["label"].values().numel())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=bs)
        for k, v in metrics.items():
            self.log(f"train_{k}", v, on_epoch=True, sync_dist=True,
                     batch_size=bs)
        return loss

    def validation_step(self, batch, _):
        output = self(batch)
        loss, metrics = self.task.loss(output, batch)
        bs = int(batch["label"].values().numel())
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=bs)
        for k, v in metrics.items():
            self.log(f"val_{k}", v, on_epoch=True, sync_dist=True, batch_size=bs)
        c = self.task.val_step_cache(output, batch)
        if c is not None:
            self._val_cache.append(c)

    def on_validation_epoch_end(self):
        if not self._val_cache:
            return
        # each rank reduces its own val shard; sync_dist averages the ranks
        for k, v in self.task.val_epoch_metrics(self._val_cache).items():
            self.log(k, v, sync_dist=True)
        self._val_cache.clear()

    def configure_optimizers(self):
        # Adam + ReduceLROnPlateau on the epoch val metric (monitored below).
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.lr_patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss_epoch",
                "interval": "epoch",
                "frequency": 1,
            },
        }
