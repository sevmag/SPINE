"""Validation callbacks for the CURTAIN pretext.

Epoch-global metrics cannot go through per-batch `self.log` averaging (a
ranking metric over the val set is not a mean of per-batch values), so they
are computed by callbacks that cache per-batch pieces and reduce once per
epoch. Enable via the run config's callback list.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from spine.pretext.curtain.task import real_query_mask
from spine.utils import auc


class CurtainValAUC(Callback):
    """Occupancy AUCs over the full validation set, split by query difficulty.

    Logs `val_auc_all`, `val_auc_hard` (positives vs nearest-dark negatives --
    the discriminative regime at the light front) and `val_auc_easy`
    (positives vs random dark sensors, which saturates quickly). Under DDP
    each rank reduces its own val shard and `sync_dist` averages the ranks.
    """

    def __init__(self):
        """Start with an empty per-epoch cache."""
        self._cache: list = []

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Drop caches of a previous epoch.

        Args:
            trainer: The running Trainer.
            pl_module: The training module.
        """
        self._cache.clear()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Cache this batch's occupancy scores, labels and difficulty tags.

        Args:
            trainer: The running Trainer.
            pl_module: The training module; its task locates the occupancy
                objective among the outputs.
            outputs: validation_step's return -- one prediction tensor per
                objective.
            batch: The collated batch.
            batch_idx: Index of the batch (unused).
            dataloader_idx: Index of the dataloader (unused).
        """
        occ = next(
            (
                i
                for i, o in enumerate(pl_module.task.objectives)
                if o.name == "occupancy"
            ),
            None,
        )
        if occ is None or outputs is None:
            return
        pred = outputs[occ]
        m = real_query_mask(pred, batch)
        self._cache.append(
            (
                pred[m].squeeze(-1).detach().float().cpu().numpy(),
                batch["label"].values().cpu().numpy(),
                batch["hard"].values().cpu().numpy() > 0.5,
            )
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reduce the cached batches and log the three AUCs.

        Args:
            trainer: The running Trainer.
            pl_module: The training module (used for logging).
        """
        if not self._cache:
            return
        lg = np.concatenate([c[0] for c in self._cache])
        y = np.concatenate([c[1] for c in self._cache])
        hd = np.concatenate([c[2] for c in self._cache])
        easy = (y == 1) | (~hd)  # positives + random (easy) negatives
        pl_module.log("val_auc_all", auc(lg, y), sync_dist=True)
        pl_module.log("val_auc_hard", auc(lg[hd], y[hd]), sync_dist=True)
        pl_module.log("val_auc_easy", auc(lg[easy], y[easy]), sync_dist=True)
        self._cache.clear()
