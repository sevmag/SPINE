"""Runnable CURTAIN pretraining example -- copy and adapt.

Builds a read Dataset (SqliteRawDataset here; swap for GraphNetRawDataset over a
graphnet LMDBDataset -- see readers.py), a CurtainTask, and the DeepIce backbone,
then calls spine.train.fit. `--task-objectives occupancy` is v1; add `,dt` for
v2 -- one flag, not a forked script. Pre-filter your selection to usable events
first (selection.filter_by_min_count).

Run:  PYTHONPATH=../src python train_curtain.py --db ... --geo ... --selection ... --out ...
"""

from __future__ import annotations

import argparse

import deepice_backbone  # noqa: F401  (registers the 'deepice' backbone)
from readers import SqliteRawDataset
from spine import train as spine_train
from spine.backbones.registry import BACKBONES
from spine.data.geometry import load_geometry
from spine.data.scaling import HexagonScaler
from spine.data.selection import load_event_nos, make_split, train_events
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
    task = CurtainTask(geo=geo, objectives=objectives, scaler=HexagonScaler())

    # TODO: pre-filter to usable events -- selection.filter_by_min_count(pool, counts, k)
    train_raw = SqliteRawDataset(args.db, train_events(split, args.n_train))
    val_raw = SqliteRawDataset(args.db, split.val)

    backbone = BACKBONES.build("deepice", d_model=128)
    spine_train.fit(train_raw, val_raw, task, backbone, args.out,
                    batch=args.batch, lr=args.lr, devices=args.devices,
                    max_epochs=args.max_epochs, patience=args.patience,
                    config=vars(args))


if __name__ == "__main__":
    main()
