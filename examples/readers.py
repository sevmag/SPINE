"""Reference read Datasets for SPINE -- NOT part of the package.

A SPINE read Dataset yields, per positional index:
    raw[i] -> {"event_no": int, "pulses": [P, 5] raw, "sensor_key": [P] int}

Copy/adapt these. GraphNeT's LMDBDataset / SQLiteDataset are the recommended
readers (fast, maintained); `GraphNetRawDataset` is the thin adapter to the
contract. SPINE standardizes AFTER the pretext split, so the read must return
RAW, unstandardized pulses.
"""

from __future__ import annotations

import sqlite3

import numpy as np
from torch.utils.data import Dataset

_PULSE_SQL = (
    "SELECT sensor_pos_x,sensor_pos_y,sensor_pos_z,t,charge{key_col} "
    "FROM {pulsemap} WHERE event_no=?"
)


class SqliteRawDataset(Dataset):
    """Minimal SQLite reference reader (per-worker lazy connection)."""

    def __init__(
        self,
        db: str,
        event_nos: np.ndarray | list[int],
        pulsemap: str = "merged_photons",
        sensor_key: str | None = "pmt_id",
    ):
        """Bind the reader to a database and an event list.

        Args:
            db: Path to the SQLite file (opened read-only, per worker).
            event_nos: The events this dataset serves, in order.
            pulsemap: Pulse table to read from.
            sensor_key: Column with the per-pulse sensor id; None reads no
                ids (sensors are then matched by coordinates).
        """
        self.db = db
        self.ev = np.asarray(event_nos)
        self.sensor_key = sensor_key
        self.sql = _PULSE_SQL.format(
            pulsemap=pulsemap,
            key_col=f",{sensor_key}" if sensor_key else "",
        )
        self._con = None

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_con"] = None  # a live SQLite handle can't cross the worker fork
        return s

    def _cur(self):
        if self._con is None:
            self._con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        return self._con.cursor()

    def __len__(self) -> int:
        return len(self.ev)

    def __getitem__(self, idx: int):
        ev = int(self.ev[idx])
        rows = np.asarray(self._cur().execute(self.sql, (ev,)).fetchall())
        if self.sensor_key is None:
            return {"event_no": ev, "pulses": rows.astype(np.float32)}
        return {
            "event_no": ev,
            "pulses": rows[:, :-1].astype(np.float32),
            "sensor_key": rows[:, -1].astype(np.int64),
        }


class GraphNetRawDataset(Dataset):
    """Adapt a graphnet Dataset (LMDBDataset / SQLiteDataset) to the contract.

    Construct the graphnet Dataset with an IDENTITY detector + `NodesAsPulses`
    so node features stay RAW and in (x, y, z, t, charge) order, and with no
    truth/labels (SPINE needs none, and standardizes after the split). This maps
    each returned `torch_geometric.Data` to {"event_no", "pulses"}.

    Example (schematic -- confirm feature order for your files):
        from graphnet.data.dataset.lmdb import LMDBDataset
        from graphnet.models.data_representation.graphs import EdgelessGraph
        from graphnet.models.data_representation.graphs.nodes import NodesAsPulses
        gn = LMDBDataset(path, pulsemaps=["merged_photons"],
                         features=["sensor_pos_x","sensor_pos_y","sensor_pos_z","t","charge"],
                         truth=[], graph_definition=EdgelessGraph(
                             detector=IdentityDetector(),
                             node_definition=NodesAsPulses()))
        reader = GraphNetRawDataset(gn)
    """

    def __init__(
        self,
        gn_dataset: Dataset,
        event_no_key: str = "event_no",
        sensor_key_index: int | None = None,
    ):
        """Wrap a graphnet Dataset.

        Args:
            gn_dataset: The graphnet Dataset to adapt (raw features, no
                truth); include the sensor-id column among its features.
            event_no_key: Attribute on each Data carrying the event id.
            sensor_key_index: Which feature column holds the sensor id; it is
                split out of the pulses into `sensor_key`.
        """
        self.ds = gn_dataset
        self.event_no_key = event_no_key
        self.sensor_key_index = sensor_key_index

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        data = self.ds[idx]  # torch_geometric Data; data.x = raw features
        x = np.asarray(data.x)
        ev = int(np.asarray(getattr(data, self.event_no_key)).reshape(-1)[0])
        if self.sensor_key_index is None:
            return {"event_no": ev, "pulses": x.astype(np.float32)}
        j = self.sensor_key_index
        feat = np.delete(x, j, axis=1).astype(np.float32)
        return {
            "event_no": ev,
            "pulses": feat,
            "sensor_key": x[:, j].astype(np.int64),
        }
