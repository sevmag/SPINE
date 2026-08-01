"""Export the backbone in transfer format on best val -- the repo's artifact.

This is the boundary to downstream finetuning: on each val-loss improvement it
writes {"backbone", "full_state", "config", "step", "val_loss"}. The finetuning
bench loads ckpt["backbone"] into its DeepIce, so the backbone state_dict keys
must stay compatible with the finetuning-side encoder.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC = P(score[pos] > score[neg]); ties ignored."""
    labels = labels.astype(bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class TransferCheckpoint(Callback):
    """Save the backbone (+ full module) when `val_loss_epoch` improves.

    Only rank 0 writes under DDP. `backbone_attr`/`full_attr` name the attributes
    on the LightningModule holding the encoder and the full pretext model.
    """

    def __init__(self, out: str, config: dict, backbone_attr: str = "backbone",
                 full_attr: str = "model", min_delta: float = 1e-4):
        self.out = out
        self.config = config
        self.backbone_attr = backbone_attr
        self.full_attr = full_attr
        self.min_delta = min_delta
        self.best = float("inf")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        vl = trainer.callback_metrics.get("val_loss_epoch")
        if vl is None:
            return
        vl = float(vl)
        if vl < self.best - self.min_delta:
            self.best = vl
            backbone = getattr(pl_module, self.backbone_attr)
            if hasattr(backbone, "transfer_module"):
                backbone = backbone.transfer_module()
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
