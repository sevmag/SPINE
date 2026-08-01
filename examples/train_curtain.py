"""Runnable CURTAIN pretraining example -- copy and adapt.

You bring the selections: `--train-selection` and `--val-selection` are event_no
parquets you create -- disjoint from each other and from your held-out test set.
SPINE owns no split; producing clean, disjoint selections is your job (a leak
inflates every pretrained-vs-scratch number). The launcher loads them, builds
read Datasets (SqliteRawDataset here; swap for GraphNetRawDataset over a graphnet
LMDBDataset -- see readers.py), a CurtainTask, and the DeepIce backbone, then
calls spine.train.fit. `--task-objectives occupancy` is v1; add `,dt` for v2 --
one flag, not a forked script. Pre-filter your selections to usable events first
(>= min_visible + min_future hit sensors), or __getitem__ fails loud.

Run:  PYTHONPATH=../src python train_curtain.py \
          --db ... --geo ... --train-selection ... --val-selection ... --out ...
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq

import deepice_backbone  # noqa: F401  (registers the 'deepice' backbone)
from readers import SqliteRawDataset
from spine import train as spine_train
from spine.backbones.registry import BACKBONES
from spine.data.geometry import load_geometry
from spine.data.scaling import HexagonScaler
from spine.pretext.curtain.objectives import OCCUPANCY, dt_objective
from spine.pretext.curtain.task import CurtainTask


def load_event_nos(path: str, column: str = "event_no"):
    """event_no array from a selection parquet you built."""
    return pq.read_table(path, columns=[column])[column].to_numpy()


def build_objectives(names, lambda_dt=1.0):
    objs = [OCCUPANCY]
    if "dt" in names:
        objs.append(dt_objective(weight=lambda_dt))
    return objs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--geo", required=True)
    p.add_argument("--train-selection", required=True,
                   help="event_no parquet for training (you build it; disjoint "
                        "from val and your test set)")
    p.add_argument("--val-selection", required=True,
                   help="event_no parquet for validation")
    p.add_argument("--n-train", type=int, default=0,
                   help="optional: use only the first n_train train events "
                        "(0 = all); the selection order defines the prefix")
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
    train_ev = load_event_nos(args.train_selection)
    if args.n_train:
        train_ev = train_ev[:args.n_train]
    val_ev = load_event_nos(args.val_selection)

    objectives = build_objectives(args.task_objectives.split(","), args.lambda_dt)
    task = CurtainTask(geo=geo, objectives=objectives, scaler=HexagonScaler())

    train_raw = SqliteRawDataset(args.db, train_ev)
    val_raw = SqliteRawDataset(args.db, val_ev)

    backbone = BACKBONES.build("deepice", d_model=128)
    spine_train.fit(train_raw, val_raw, task, backbone, args.out,
                    batch=args.batch, lr=args.lr, devices=args.devices,
                    max_epochs=args.max_epochs, patience=args.patience,
                    config=vars(args))


if __name__ == "__main__":
    main()
