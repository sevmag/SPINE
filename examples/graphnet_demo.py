"""End-to-end CURTAIN pretraining on graphnet's bundled Prometheus mock data.

Demonstrates the graphnet frame around the spine core using the 50-event demo
file shipped inside a graphnet checkout: a graphnet SQLiteDataset is adapted
to the RawEvent contract, the geometry asset and the event selection are built
from the file itself, a tiny DeepIce is pretrained (about a minute on CPU),
and the exported checkpoint is loaded back into a stock graphnet DeepIce ready
for supervised fine-tuning.

Usage (spine installed per the README, graphnet checkout on the import path):
    python examples/graphnet_demo.py --out curtain_demo_out
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from graphnet.constants import EXAMPLE_DATA_DIR
from graphnet.data.dataset import SQLiteDataset
from graphnet.models.data_representation import EdgelessGraph, NodesAsPulses
from graphnet.models.detector import Detector
from graphnet.models.gnn import DeepIce
from spine_graphnet.deepice_backbone import DeepIceBackbone
from spine_graphnet.readers import GraphNetRawDataset
from torch.utils.data import Dataset

from spine.data.geometry import load_geometry
from spine.data.scaling import FeatureLayout, FeatureScaler
from spine.pretext.curtain.callbacks import CurtainValAUC
from spine.pretext.curtain.objectives import OccupancyObjective
from spine.pretext.curtain.sampler import can_always_split
from spine.pretext.curtain.task import CurtainTask
from spine.train import fit

# raw columns read from the demo file; sensor_id maps 1:1 to a position there,
# so it doubles as the data-carried sensor key
FEATURES = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", "sensor_id"]
LAYOUT = FeatureLayout()

# the selection filter guarantees make_sample cannot raise only under the very
# same knob values the task later samples with -- single-source them
SPLIT_KNOBS = {
    "min_visible": 8,
    "min_future": 4,
}

TINY_BACKBONE = {
    "d_model": 32,
    "depth": 2,
    "head_size": 8,
    "depth_rel": 1,
    "n_rel": 1,
    "seq_length": 32,
}


def _identity(x: torch.Tensor) -> torch.Tensor:
    """Return the input unchanged.

    Args:
        x: Feature column values.

    Returns:
        The same values.
    """
    return x


class RawPrometheus(Detector):
    """Identity detector: features stay raw; spine scales after sampling."""

    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> dict[str, Callable]:
        """Map every read column to the identity.

        Returns:
            Identity standardization for each column in FEATURES.
        """
        return {name: _identity for name in FEATURES}


class UnitCharge(Dataset):
    """Append charge = 1 to each pulse row.

    Prometheus demo rows are single photons with no charge column; unit charge
    per row completes the (x, y, z, t, charge) layout and turns the sampler's
    charge-weighted mean time into the plain mean of the visible photon times.
    """

    def __init__(self, raw: Dataset):
        """Wrap a RawEvent dataset whose pulses lack the charge column.

        Args:
            raw: Dataset yielding RawEvents with (x, y, z, t) pulses.
        """
        self.raw = raw

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> dict:
        ev = dict(self.raw[idx])
        p = ev["pulses"]
        ev["pulses"] = np.concatenate([p, np.ones((len(p), 1), np.float32)], axis=1)
        return ev


class PrometheusDemoScaler(FeatureScaler):
    """Demo-detector scaling: positions span about 100 m, times about 1 us."""

    def scale_pulses(self, x: torch.Tensor) -> torch.Tensor:
        """Scale xyz and t to O(1); the unit charge passes through.

        Args:
            x: [..., F] raw pulse features, columns per `self.layout`.

        Returns:
            Standardized features, same shape and column order.
        """
        lay = self.layout
        out = x.clone()
        out[..., list(lay.pos)] = x[..., list(lay.pos)] / 100.0
        out[..., lay.t] = x[..., lay.t] / 1000.0
        return out

    def scale_positions(self, p: torch.Tensor) -> torch.Tensor:
        """Scale raw positions with the same factor as the pulse xyz.

        Args:
            p: [..., 3] raw positions in metres.

        Returns:
            Standardized coordinates on the same scale as scaled pulse xyz.
        """
        return p / 100.0


def build_geometry_asset(db: str, out_path: str) -> None:
    """Write the per-sensor geometry asset derived from the demo file.

    The demo detector is taken to be every sensor appearing in the file.
    Neighbour lists cover all other sensors sorted by distance, so the
    sampler's nearest-dark lookup always finds a dark sensor.

    Args:
        db: Path to the demo SQLite file.
        out_path: Destination .npz path.
    """
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT DISTINCT sensor_id, sensor_pos_x, sensor_pos_y, sensor_pos_z "
        "FROM total ORDER BY sensor_id"
    ).fetchall()
    con.close()
    arr = np.asarray(rows, dtype=np.float64)
    xyz = arr[:, 1:].astype(np.float32)
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    np.savez(
        out_path,
        xyz=xyz,
        knn_idx=np.argsort(dist, axis=1)[:, 1:],
        sensor_id=arr[:, 0].astype(np.int64),
    )


def make_raw_dataset(db: str, selection: list[int] | None = None) -> Dataset:
    """Adapt a graphnet SQLiteDataset to the RawEvent contract.

    The identity detector plus NodesAsPulses keeps node features raw and in
    FEATURES order; truth stays empty because pretraining needs no labels.

    Args:
        db: Path to the demo SQLite file.
        selection: Event numbers to read; None reads all.

    Returns:
        RawEvent dataset with (x, y, z, t, charge) pulses and sensor keys.
    """
    gn = SQLiteDataset(
        path=db,
        pulsemaps=["total"],
        features=FEATURES,
        truth=[],
        truth_table="mc_truth",
        data_representation=EdgelessGraph(
            detector=RawPrometheus(),
            node_definition=NodesAsPulses(),
            input_feature_names=FEATURES,
        ),
        selection=selection,
    )
    return UnitCharge(
        GraphNetRawDataset(gn, sensor_key_index=FEATURES.index("sensor_id"))
    )


def build_selection(
    raw: Dataset, val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Filter to guaranteed-splittable events and split into train/val.

    Args:
        raw: RawEvent dataset over the full file.
        val_frac: Fraction of the kept events used for validation.
        seed: Shuffle seed.

    Returns:
        Train and validation event-number lists.
    """
    keep = []
    for i in range(len(raw)):
        ev = raw[i]
        pt = ev["pulses"][:, LAYOUT.t]
        if can_always_split(pt, ev["sensor_key"], **SPLIT_KNOBS):
            keep.append(ev["event_no"])
    order = np.random.default_rng(seed).permutation(len(keep))
    n_val = max(1, round(val_frac * len(keep)))
    val = [keep[i] for i in order[:n_val]]
    train = [keep[i] for i in order[n_val:]]
    return train, val


def parse_args() -> argparse.Namespace:
    """Parse the demo's command line.

    Returns:
        Parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        default=None,
        help="demo SQLite file; default: the graphnet checkout's bundled copy",
    )
    ap.add_argument("--out", default="curtain_demo_out", help="output directory")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def main() -> None:
    """Run the demo end to end.

    Raises:
        SystemExit: If the demo data is absent and no --db was given.
    """
    args = parse_args()
    db = args.db or os.path.join(
        EXAMPLE_DATA_DIR, "sqlite", "prometheus", "prometheus-events.db"
    )
    if not os.path.exists(db):
        raise SystemExit(
            f"demo data not found at {db} -- clone graphnet (the file ships in "
            "its repo under data/examples) or pass --db"
        )
    os.makedirs(args.out, exist_ok=True)

    geo_path = os.path.join(args.out, "prometheus_demo_geometry.npz")
    build_geometry_asset(db, geo_path)
    geo = load_geometry(geo_path, sensor_key="sensor_id")

    train_ev, val_ev = build_selection(
        make_raw_dataset(db), val_frac=0.2, seed=args.seed
    )
    print(
        f"selection: {len(train_ev)} train / {len(val_ev)} val "
        "guaranteed-splittable events",
        flush=True,
    )

    task = CurtainTask(
        geo=geo,
        # v2 is one line more: append DtObjective(weight=1.0) from
        # spine.pretext.curtain.objectives
        objectives=[OccupancyObjective()],
        scaler=PrometheusDemoScaler(),
        dt_scale=100.0,
        **SPLIT_KNOBS,
    )
    backbone = DeepIceBackbone(**TINY_BACKBONE)
    ckpt_path = os.path.join(args.out, "curtain_prometheus_demo.pth")
    fit(
        make_raw_dataset(db, selection=train_ev),
        make_raw_dataset(db, selection=val_ev),
        task,
        backbone,
        ckpt_path,
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-4),
        batch=8,
        num_workers=0,
        devices=1,
        max_epochs=args.epochs,
        patience=10,
        callbacks=[CurtainValAUC()],
    )

    # the payoff of the graphnet connection: the exported backbone weights
    # drop straight into a stock graphnet DeepIce for supervised fine-tuning
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stock = DeepIce(
        hidden_dim=TINY_BACKBONE["d_model"],
        depth=TINY_BACKBONE["depth"],
        head_size=TINY_BACKBONE["head_size"],
        depth_rel=TINY_BACKBONE["depth_rel"],
        n_rel=TINY_BACKBONE["n_rel"],
        seq_length=TINY_BACKBONE["seq_length"],
        include_dynedge=False,
        n_features=5,
    )
    stock.load_state_dict(ckpt["backbone"])
    print(
        f"best val loss {ckpt['val_loss']:.4f} (step {ckpt['step']}); "
        f"backbone from {ckpt_path} loaded into a stock graphnet DeepIce",
        flush=True,
    )


if __name__ == "__main__":
    main()
