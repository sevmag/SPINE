"""Transfer-checkpoint export -- the artifact this repo exists to produce."""

from __future__ import annotations

import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class TransferCheckpoint(Callback):
    """Save the backbone (+ full module) when `val_loss_epoch` improves.

    Only rank 0 writes under DDP; reads `pl_module.backbone` (the exported
    encoder) and `pl_module.model` (the full pretext model).
    """

    def __init__(self, out: str, config: dict, min_delta: float = 1e-4):
        """Configure the export target.

        Args:
            out: Checkpoint path; parent directories are created.
            config: Run configuration stored inside the checkpoint.
            min_delta: Required val-loss improvement before re-exporting.
        """
        self.out = out
        self.config = config
        self.min_delta = min_delta
        self.best = float("inf")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Export the transfer checkpoint when the epoch val loss improves.

        Args:
            trainer: The running Trainer (rank and logged metrics).
            pl_module: The LightningModule carrying backbone and full model.
        """
        if trainer.global_rank != 0:
            return
        vl = trainer.callback_metrics.get("val_loss_epoch")
        if vl is None:
            return
        vl = float(vl)
        if vl < self.best - self.min_delta:
            self.best = vl
            torch.save(
                {
                    "backbone": pl_module.backbone.state_dict(),
                    "full_state": pl_module.model.state_dict(),
                    "config": self.config,
                    "step": trainer.global_step,
                    "val_loss": vl,
                },
                self.out,
            )
