"""Optimizer + scheduler: Adam + ReduceLROnPlateau on the epoch val metric."""

from __future__ import annotations

import torch


def build_optimizer(params, lr: float, lr_patience: int, factor: float = 0.5):
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=factor, patience=lr_patience
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
