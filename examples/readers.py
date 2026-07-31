"""Reference read Datasets for SPINE -- NOT part of the package.

A SPINE read Dataset yields, per positional index:
    raw[i] -> {"event_no": int, "pulses": np.ndarray[P, 5]}   # x,y,z,t,charge RAW

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
    "SELECT sensor_pos_x,sensor_pos_y,sensor_pos_z,t,charge "
    "FROM {pulsemap} WHERE event_no=?"
)


class SqliteRawDataset(Dataset):
    """Minimal SQLite reference reader (per-worker lazy connection)."""

    def __init__(self, db: str, event_nos, pulsemap: str = "merged_photons"):
        self.db = db
        self.ev = np.asarray(event_nos)
        self.sql = _PULSE_SQL.format(pulsemap=pulsemap)
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
        rows = self._cur().execute(self.sql, (ev,)).fetchall()
        return {"event_no": ev, "pulses": np.asarray(rows, np.float32)}


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
                             detector=IdentityDetector(), node_definition=NodesAsPulses()))
        reader = GraphNetRawDataset(gn)
    """

    def __init__(self, gn_dataset, event_no_key: str = "event_no"):
        self.ds = gn_dataset
        self.event_no_key = event_no_key

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        data = self.ds[idx]  # torch_geometric Data; data.x = [P,5] raw features
        pulses = np.asarray(data.x, dtype=np.float32)
        ev = int(np.asarray(getattr(data, self.event_no_key)).reshape(-1)[0])
        return {"event_no": ev, "pulses": pulses}
