"""Raw-pulse read layer -- a plain PyTorch Dataset (index -> raw pulses).

`raw[i] -> {"event_no", "pulses": np.ndarray[P,5]}` for the i-th event of a
selection. The pretext transform composes on top (PretextDataset), so SQLite vs
LMDB is just which read Dataset you build -- no bespoke read interface. The read
returns RAW pulses; standardization happens after the pretext split.

`SqliteRawDataset` is the current path. For LMDB, use graphnet's `LMDBDataset`
(itself a Dataset) returning raw pulses, or a thin adapter yielding the same
{"event_no", "pulses"} dict -- both slot straight into PretextDataset.
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
    """Index -> raw pulses of the i-th selected event (per-worker lazy conn)."""

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
