"""Export the backbone in transfer format on best val -- the repo's artifact.

This is the boundary to downstream finetuning: on each val-loss improvement it
writes {"backbone", "full_state", "config", "step", "val_loss"}. The finetuning
bench loads ckpt["backbone"] into its DeepIce, so the backbone state_dict keys
must stay compatible with the finetuning-side encoder.
"""

from __future__ import annotations

import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class TransferCheckpoint(Callback):
    """Save the backbone (+ full module) when `val_loss_epoch` improves.

    Only rank 0 writes under DDP. `backbone_attr`/`full_attr` name the attributes
    on the LightningModule holding the encoder and the full pretext model.
    """

    def __init__(
        self,
        out: str,
        config: dict,
        backbone_attr: str = "backbone",
        full_attr: str = "model",
        min_delta: float = 1e-4,
    ):
        """Configure the export target and what to read off the module.

        Args:
            out: Checkpoint path; parent directories are created.
            config: Run configuration stored inside the checkpoint.
            backbone_attr: LightningModule attribute holding the encoder.
            full_attr: LightningModule attribute holding the full model.
            min_delta: Required val-loss improvement before re-exporting.
        """
        self.out = out
        self.config = config
        self.backbone_attr = backbone_attr
        self.full_attr = full_attr
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
            backbone = getattr(pl_module, self.backbone_attr)
            full = getattr(pl_module, self.full_attr, backbone)
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "full_state": full.state_dict(),
                    "config": self.config,
                    "step": trainer.global_step,
                    "val_loss": vl,
                },
                self.out,
            )
