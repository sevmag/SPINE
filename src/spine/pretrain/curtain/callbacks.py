"""Validation callbacks for the CURTAIN pretrain.

Epoch-global metrics (AUC is rank-based over the full val set) cannot flow
through per-batch log averaging, so callbacks cache per batch and reduce once
per epoch. Enable via the run config's callback list.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from spine.pretrain.curtain.task import real_query_mask


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score of a positive > score of a negative).

    Rank-based and threshold-free, so it measures discrimination independent
    of calibration and of the positive/negative balance. Ties are ignored.

    Args:
        scores: Per-item scores (higher = more positive).
        labels: Per-item binary labels (nonzero = positive).

    Returns:
        The AUC, or nan when either class is absent.
    """
    labels = labels.astype(bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


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
        outputs: list | None,
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


class CurtainValLossMedian(Callback):
    """Per-event median (and IQR) of the validation loss.

    The mean CURTAIN loss is dominated by a few pathological events, which
    makes the epoch-to-epoch curve noisy and, with early stopping, makes the
    chosen epoch partly a matter of luck. The median over events is far more
    stable at the same cost, so it is logged alongside the mean as
    `val_loss_median` (plus `val_loss_p25`/`val_loss_p75` for spread).

    Per-event loss is the mean occupancy BCE over that event's real queries;
    for the v1 (occupancy-only) objective set that is exactly the loss being
    optimized. Logged for monitoring only -- nothing selects on it unless a
    run points EarlyStopping/checkpointing at it.
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
        outputs: list | None,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Cache this batch's per-event mean occupancy loss.

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
        per_query = F.binary_cross_entropy_with_logits(
            pred[m].squeeze(-1).float(),
            batch["label"].values().float(),
            reduction="none",
        )
        # queries pack per event; the NJT offsets give each event's slice
        counts = batch["qpos"].offsets().diff()
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
        per_event = torch.stack(
            [per_query[s:e].mean() for s, e in zip(starts, ends, strict=True) if e > s]
        )
        self._cache.append(per_event.detach().cpu().numpy())

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log the median and quartiles over this rank's validation events.

        Args:
            trainer: The running Trainer.
            pl_module: The training module (used for logging).
        """
        if not self._cache:
            return
        v = np.concatenate(self._cache)
        # sync_dist averages the ranks' medians -- an approximation of the
        # global median, which is fine for a monitoring statistic
        pl_module.log("val_loss_median", float(np.median(v)), sync_dist=True)
        pl_module.log("val_loss_p25", float(np.percentile(v, 25)), sync_dist=True)
        pl_module.log("val_loss_p75", float(np.percentile(v, 75)), sync_dist=True)
        self._cache.clear()
