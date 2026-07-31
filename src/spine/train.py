"""SPINE training entrypoint: build backbone + task + data -> Trainer.fit.

Skeleton wiring. TODO: replace argparse with the config system (hydra) and load
objectives/sampler/detector from config groups. `--task-objectives` selects v1
(occupancy) vs v2 (occupancy,dt) -- one flag, not a forked script.

NB: the datamodule now assumes a *pre-filtered* selection (usable events only).
Filter train/val event_nos with `selection.filter_by_min_count(...)` before
passing them in -- __getitem__ fails loud on an unusable event.
"""

from __future__ import annotations

import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from spine.backbones.registry import BACKBONES
from spine.data.datamodule import SpineDataModule
from spine.data.geometry import load_geometry
from spine.data.selection import load_event_nos, make_split, train_events
from spine.data.sources import SqliteRawDataset
from spine.engine.module import SSLModule
from spine.engine.transfer import TransferCheckpoint
from spine.pretext.curtain.objectives import OCCUPANCY, dt_objective
from spine.pretext.curtain.task import CurtainTask


def build_objectives(names, lambda_dt=1.0):
    objs = [OCCUPANCY]
    if "dt" in names:
        objs.append(dt_objective(weight=lambda_dt))
    return objs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--geo", required=True)
    p.add_argument("--selection", required=True, help="ordered event_no parquet")
    p.add_argument("--n-train", type=int, default=100_000)
    p.add_argument("--task-objectives", default="occupancy",
                   help="comma list: occupancy[,dt]  (v1 vs v2)")
    p.add_argument("--lambda-dt", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    geo = load_geometry(args.geo)
    split = make_split(load_event_nos(args.selection))
    objectives = build_objectives(args.task_objectives.split(","), args.lambda_dt)
    task = CurtainTask(geo=geo, objectives=objectives)

    # TODO: filter to usable events -- selection.filter_by_min_count(pool, counts, k)
    train_raw = SqliteRawDataset(args.db, train_events(split, args.n_train))
    val_raw = SqliteRawDataset(args.db, split.val)
    dm = SpineDataModule(train_raw, val_raw, task, batch_size=args.batch)

    backbone = BACKBONES.build("deepice", d_model=128)
    module = SSLModule(backbone, task, lr=args.lr)

    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        precision="32-true",
        max_epochs=args.max_epochs,
        callbacks=[
            TransferCheckpoint(args.out, config=vars(args)),
            EarlyStopping(monitor="val_loss_epoch", mode="min",
                          patience=args.patience),
        ],
    )
    trainer.fit(module, datamodule=dm)


if __name__ == "__main__":
    main()
