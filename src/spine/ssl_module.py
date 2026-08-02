"""Lightning module wiring a Backbone to a PretextTask's head.

Logs with sync_dist=True so ReduceLROnPlateau steps identically on every DDP
rank; exposes `backbone` and `model` for TransferCheckpoint. Batch access
follows the CURTAIN collate contract (qpos / label).
"""

from __future__ import annotations

from collections.abc import Callable

import pytorch_lightning as pl
from torch import nn

from spine.backbones.base import Backbone
from spine.pretext.base import PretextTask


class SSLModule(pl.LightningModule):
    """Backbone + pretext head; the task computes the loss."""

    def __init__(
        self,
        backbone: Backbone,
        task: PretextTask,
        optimizer: Callable,
        scheduler: Callable | None = None,
        scheduler_config: dict | None = None,
    ):
        """Wire backbone + head and store the optimization factories.

        Args:
            backbone: Encoder to pretrain.
            task: Pretext task; builds the head and computes the loss.
            optimizer: Factory, parameters -> torch Optimizer (partial).
            scheduler: Optional factory, optimizer -> LR scheduler.
            scheduler_config: Lightning lr_scheduler metadata; None means
                epoch-level plateau on the val loss, a provided dict REPLACES
                the default (no stale `monitor` for step-based schedulers).
        """
        super().__init__()
        self.task = task
        self.model = nn.ModuleDict(
            {"backbone": backbone, "head": task.build_head(backbone.out_dim)}
        )
        self.opt_factory = optimizer
        self.sched_factory = scheduler
        self.sched_config = (
            dict(scheduler_config)
            if scheduler_config
            else {"monitor": "val_loss_epoch", "interval": "epoch", "frequency": 1}
        )

    @property
    def backbone(self) -> nn.Module:
        """The encoder -- the module the transfer checkpoint exports."""
        return self.model["backbone"]

    @property
    def head(self) -> nn.Module:
        """The pretext prediction head."""
        return self.model["head"]

    def forward(self, batch: dict):
        """Encode the batch and run the head on the padded query positions.

        Args:
            batch: The task's collated batch.

        Returns:
            The head's output for the batch.
        """
        enc = self.backbone.encode(batch)
        return self.head(batch["qpos"].to_padded_tensor(0.0), enc)

    def _step(self, batch, stage: str):
        output = self(batch)
        loss, metrics = self.task.loss(output, batch)
        bs = int(batch["label"].values().numel())
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=True,  # both flags -> Lightning emits the _epoch-suffixed metric,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )
        for k, v in metrics.items():
            self.log(f"{stage}_{k}", v, on_epoch=True, sync_dist=True, batch_size=bs)
        return output, loss

    def training_step(self, batch: dict, _) -> object:
        """Compute, log and return the training loss.

        Args:
            batch: The task's collated batch.

        Returns:
            The loss tensor Lightning backpropagates.
        """
        return self._step(batch, "train")[1]

    def validation_step(self, batch: dict, _) -> object:
        """Compute and log the validation loss.

        Args:
            batch: The task's collated batch.

        Returns:
            The model output, so callbacks can consume it
            (on_validation_batch_end receives it as `outputs`).
        """
        return self._step(batch, "val")[0]

    def configure_optimizers(self):
        """Build the optimizer (and scheduler bundle) from the factories.

        Returns:
            The optimizer alone, or Lightning's optimizer/lr_scheduler dict
            when a scheduler factory was given.
        """
        opt = self.opt_factory(self.parameters())
        if self.sched_factory is None:
            return opt
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": self.sched_factory(opt), **self.sched_config},
        }
