"""SPINE run assembly (reader-agnostic).

`fit()` wires read Datasets + a pretext task + a backbone into the SSLModule and
Trainer (transfer-checkpoint export + early stopping) and fits. It owns none of
the data reading: pass any Datasets satisfying the RawPulseDataset contract
(`raw[i] -> {"event_no", "pulses": [P,5] raw}`). Build one with graphnet's
LMDBDataset / SQLiteDataset -- see examples/readers.py. A runnable CURTAIN
launcher is examples/train_curtain.py.
"""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import Dataset

from spine.backbones.base import Backbone
from spine.data.datamodule import SpineDataModule
from spine.ssl_module import SSLModule
from spine.utils import TransferCheckpoint
from spine.pretext.base import PretextTask


def fit(train_raw: Dataset, val_raw: Dataset, task: PretextTask,
        backbone: Backbone, out: str, *, batch: int = 64, lr: float = 5e-4,
        num_workers: int = 16, devices: int = 1, precision: str = "32-true",
        max_epochs: int = 200, patience: int = 15,
        config: Optional[dict] = None):
    """Assemble and fit. Returns the trained SSLModule."""
    dm = SpineDataModule(train_raw, val_raw, task, batch_size=batch,
                         num_workers=num_workers)
    module = SSLModule(backbone, task, lr=lr)
    trainer = pl.Trainer(
        devices=devices, accelerator="gpu", precision=precision,
        max_epochs=max_epochs,
        callbacks=[
            TransferCheckpoint(out, config=config or {}),
            EarlyStopping(monitor="val_loss_epoch", mode="min", patience=patience),
        ],
    )
    trainer.fit(module, datamodule=dm)
    return module
