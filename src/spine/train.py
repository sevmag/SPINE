"""SPINE run assembly (reader-agnostic).

`fit()` wires read Datasets + a pretext task + a backbone into the SSLModule and
Trainer (transfer-checkpoint export + early stopping) and fits. It owns none of
the data reading: pass any Datasets satisfying the RawPulseDataset contract
(`raw[i] -> {"event_no", "pulses": [P,F] raw}`). Build one with graphnet's
LMDBDataset / SQLiteDataset -- see examples/readers.py. A runnable CURTAIN
launcher is examples/train_curtain.py.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

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


def fit(train_raw: Dataset, val_raw: Dataset, task: PretextTask,
        backbone: Backbone, out: str, *, batch: int = 64, lr: float = 5e-4,
        lr_patience: int = 7, num_workers: int = 16,
        val_num_workers: Optional[int] = None, devices: int = 1,
        precision: str = "32-true", max_epochs: int = 200, patience: int = 15,
        grad_clip: float = 1.0, wandb: Optional[dict] = None,
        config: Optional[dict] = None):
    """Assemble and fit. Returns the trained SSLModule.

    `wandb` (optional): {project, group, name, mode, tags} enables a
    WandbLogger + LR monitoring; None trains without a logger.
    """
    # TF32 matmuls: the fp32 recipe this repo targets trains with
    # torch.set_float32_matmul_precision("high") on Ampere+.
    torch.set_float32_matmul_precision("high")
    dm = SpineDataModule(train_raw, val_raw, task, batch_size=batch,
                         num_workers=num_workers,
                         val_num_workers=val_num_workers)
    module = SSLModule(backbone, task, lr=lr, lr_patience=lr_patience)

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
            name=wandb.get("name"), group=wandb.get("group"),
            offline=wandb.get("mode") == "offline",
            tags=list(wandb.get("tags") or []),
        )
        logger.log_hyperparams({**(config or {}), "params": n_par})
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        print(f"params={n_par / 1e6:.2f}M  wandb={wandb.get('name')}",
              flush=True)

    # broadcast_buffers=False: the only buffers are compile-time constants
    # (Fourier frequencies), and the per-forward broadcast is the collective
    # that dies when a rank stalls (e.g. a slow checkpoint write to shared
    # storage); the long timeout rides out storage hiccups.
    strategy = (DDPStrategy(cluster_environment=LightningEnvironment(),
                            broadcast_buffers=False,
                            timeout=timedelta(hours=2))
                if devices > 1 else "auto")
    trainer = pl.Trainer(
        accelerator="gpu", devices=devices, strategy=strategy,
        precision=precision, max_epochs=max_epochs,
        gradient_clip_val=grad_clip, num_sanity_val_steps=0,
        enable_checkpointing=False, log_every_n_steps=100,
        logger=logger, callbacks=callbacks,
    )
    trainer.fit(module, datamodule=dm)
    return module
