"""Run assembly: fit() builds datamodule, module and Trainer, then fits.

Reader-agnostic -- pass any Datasets satisfying the RawPulseDataset contract
(spine.data.datamodule). A runnable launcher is examples/train_curtain.py.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import timedelta

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins.environments import LightningEnvironment
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset

from spine.backbones.base import Backbone
from spine.data.datamodule import SpineDataModule
from spine.pretrain.base import PretrainTask
from spine.ssl_module import SSLModule
from spine.utils import TransferCheckpoint


def fit(
    train_raw: Dataset,
    val_raw: Dataset,
    task: PretrainTask,
    backbone: Backbone,
    out: str,
    *,
    optimizer: Callable,
    scheduler: Callable | None = None,
    scheduler_config: dict | None = None,
    batch: int = 64,
    num_workers: int = 16,
    val_num_workers: int | None = None,
    devices: int = 1,
    precision: str = "32-true",
    max_epochs: int = 200,
    patience: int = 15,
    grad_clip: float = 1.0,
    callbacks: list | None = None,
    wandb: dict | None = None,
    config: dict | None = None,
    init_from: str | None = None,
    resume_from: str | None = None,
    save_state: str | None = None,
):
    """Assemble the datamodule, module and Trainer, then fit.

    Args:
        train_raw: Read Dataset for the training events.
        val_raw: Read Dataset for the validation events.
        task: Pretrain task (sampling, collate, head, loss).
        backbone: Encoder to pretrain; its state_dict is the exported artifact.
        out: Path the transfer checkpoint is written to on best val loss.
        optimizer: Factory mapping parameters -> a torch Optimizer.
        scheduler: Optional factory mapping that optimizer -> an LR scheduler.
        scheduler_config: Lightning lr_scheduler metadata; None uses the
            plateau-on-val-loss default.
        batch: Events per batch.
        num_workers: Training-loader worker processes.
        val_num_workers: Validation-loader workers; None uses num_workers.
        devices: Accelerator devices; more than one trains with DDP.
        precision: Lightning precision string.
        max_epochs: Epoch ceiling (early stopping usually ends the run).
        patience: EarlyStopping patience in epochs on the val loss.
        grad_clip: Gradient-norm clip value.
        callbacks: Extra Lightning callbacks appended to the built-ins.
        wandb: Optional {project, group, name, mode, tags} enabling a
            WandbLogger + LR monitoring; None trains without a logger.
        config: Run configuration stored in the checkpoint and logged.
        init_from: Prior TransferCheckpoint to warm-start the pretext model
            from; weights only, no optimizer state.
        resume_from: Lightning ``last.ckpt`` to resume from with full state
            (optimizer, scheduler, callbacks, loop). Mutually exclusive
            with ``init_from``.
        save_state: Directory for the rolling full-state ``last.ckpt``,
            refreshed each validation epoch; None uses ``<out stem>_state/``.

    Returns:
        The trained SSLModule.

    Raises:
        ValueError: If both ``init_from`` and ``resume_from`` are given.
    """
    # fp32 matmuls on TF32 tensor cores: a large speedup on Ampere+ GPUs with
    # far less precision loss than bf16-mixed
    torch.set_float32_matmul_precision("high")
    dm = SpineDataModule(
        train_raw,
        val_raw,
        task,
        batch_size=batch,
        num_workers=num_workers,
        val_num_workers=val_num_workers,
    )
    module = SSLModule(
        backbone,
        task,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )
    if init_from is not None and resume_from is not None:
        raise ValueError(
            "init_from and resume_from are mutually exclusive: a full-state "
            "resume already restores the weights"
        )
    if init_from is not None:
        prior = torch.load(init_from, map_location="cpu", weights_only=False)
        module.model.load_state_dict(prior["full_state"])
        print(
            f"warm-start: loaded pretext model from {init_from} "
            f"(val_loss={prior.get('val_loss')})",
            flush=True,
        )

    # checked at validation end: a resume replays on_train_epoch_end without
    # validation metrics, where the default check would raise
    cbs = [
        TransferCheckpoint(out, config=config or {}),
        EarlyStopping(
            monitor="val_loss_epoch",
            mode="min",
            patience=patience,
            check_on_train_epoch_end=False,
        ),
        *(callbacks or []),
    ]
    if save_state is None:
        save_state = f"{os.path.splitext(out)[0]}_state"
    # monitor=None + save_top_k=1: Lightning refreshes last.ckpt only
    # alongside a top-k save; save_top_k=0 would defer it to on_train_end,
    # useless for crash/timeout recovery
    cbs.append(
        ModelCheckpoint(
            dirpath=save_state,
            monitor=None,
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
    )
    logger = False
    if wandb:
        from pytorch_lightning.loggers import WandbLogger

        n_par = sum(p.numel() for p in module.parameters())
        logger = WandbLogger(
            project=wandb.get("project", "spine"),
            name=wandb.get("name"),
            group=wandb.get("group"),
            offline=wandb.get("mode") == "offline",
            tags=list(wandb.get("tags") or []),
        )
        logger.log_hyperparams({**(config or {}), "params": n_par})
        cbs.append(LearningRateMonitor(logging_interval="step"))
        print(f"params={n_par / 1e6:.2f}M  wandb={wandb.get('name')}", flush=True)

    # broadcast_buffers off: the encoder's only buffers are constants, and the
    # per-forward buffer broadcast is the collective that hangs when one rank
    # stalls (e.g. a slow checkpoint write to shared storage); the long NCCL
    # timeout rides out such stalls instead of aborting a healthy run.
    strategy = (
        DDPStrategy(
            cluster_environment=LightningEnvironment(),
            broadcast_buffers=False,
            timeout=timedelta(hours=2),
        )
        if devices > 1
        else "auto"
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=grad_clip,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        log_every_n_steps=100,
        logger=logger,
        callbacks=cbs,
    )
    trainer.fit(module, datamodule=dm, ckpt_path=resume_from)
    return module
