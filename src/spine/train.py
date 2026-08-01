"""SPINE run assembly (reader-agnostic).

`fit()` wires read Datasets + a pretext task + a backbone into the SSLModule and
Trainer (transfer-checkpoint export + early stopping) and fits. It owns none of
the data reading: pass any Datasets satisfying the RawPulseDataset contract
(`raw[i] -> {"event_no", "pulses": [P,F] raw}`). Build one with graphnet's
LMDBDataset / SQLiteDataset -- see examples/readers.py. A runnable CURTAIN
launcher is examples/train_curtain.py.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pytorch_lightning as pl
import torch
from lightning_fabric.plugins.environments import LightningEnvironment
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset

from spine.backbones.base import Backbone
from spine.data.datamodule import SpineDataModule
from spine.pretext.base import PretextTask
from spine.ssl_module import SSLModule
from spine.utils import TransferCheckpoint


def fit(
    train_raw: Dataset,
    val_raw: Dataset,
    task: PretextTask,
    backbone: Backbone,
    out: str,
    *,
    optimizer: Callable,
    scheduler: Callable | None = None,
    scheduler_config: dict | None = None,
    batch: int = 64,
    num_workers: int = 16,
    devices: int = 1,
    precision: str = "32-true",
    max_epochs: int = 200,
    patience: int = 15,
    grad_clip: float = 1.0,
    wandb: dict | None = None,
    config: dict | None = None,
):
    """Assemble and fit. Returns the trained SSLModule.

    `optimizer` / `scheduler` are factories (parameters -> Optimizer,
    optimizer -> scheduler); see SSLModule. `wandb` (optional:
    {project, group, name, mode, tags}) enables a WandbLogger + LR monitoring;
    None trains without a logger.
    """
    # fp32 matmuls on TF32 tensor cores: a large speedup on Ampere+ GPUs with
    # far less precision loss than bf16-mixed
    torch.set_float32_matmul_precision("high")
    dm = SpineDataModule(
        train_raw, val_raw, task, batch_size=batch, num_workers=num_workers
    )
    module = SSLModule(
        backbone,
        task,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
    )

    callbacks = [
        TransferCheckpoint(out, config=config or {}),
        EarlyStopping(monitor="val_loss_epoch", mode="min", patience=patience),
    ]
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
        callbacks.append(LearningRateMonitor(logging_interval="step"))
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
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=grad_clip,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        log_every_n_steps=100,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=dm)
    return module
