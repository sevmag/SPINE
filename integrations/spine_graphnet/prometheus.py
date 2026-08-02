"""RawEvent adapters for graphnet's bundled Prometheus demo file.

The 50-event example file shipped in a graphnet checkout has per-photon rows
with no charge column and a globally unique `sensor_id` that serves as the
data-carried sensor key. `demo_reader` builds the full reader chain over it;
it backs both the plain demo script (examples/graphnet_demo.py) and the Hydra
data group (configs/data/prometheus_demo.yaml).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from graphnet.data.dataset import SQLiteDataset
from graphnet.models.data_representation import EdgelessGraph, NodesAsPulses
from graphnet.models.detector import Detector
from torch.utils.data import Dataset

from spine_graphnet.readers import GraphNetRawDataset

FEATURES = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", "sensor_id"]


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


def demo_reader(db: str, event_nos: Sequence[int] | None = None) -> Dataset:
    """Build the RawEvent reader chain over the Prometheus demo file.

    The identity detector plus NodesAsPulses keeps node features raw and in
    FEATURES order; truth stays empty because pretraining needs no labels.

    Args:
        db: Path to the demo SQLite file.
        event_nos: Event numbers to read; None reads all.

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
        selection=None if event_nos is None else [int(e) for e in event_nos],
    )
    return UnitCharge(
        GraphNetRawDataset(gn, sensor_key_index=FEATURES.index("sensor_id"))
    )
