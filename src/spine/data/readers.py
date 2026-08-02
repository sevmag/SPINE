"""Reference SQLite reader satisfying the RawPulseDataset contract.

Optional -- any Dataset meeting the contract (spine.data.datamodule) works;
this one reads raw pulses straight from a pulsemap table. graphnet-backed
readers live in the spine_graphnet integration package.
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
        sensor_key: str = "pmt_id",
    ):
        """Bind the reader to a database and an event list.

        Args:
            db: Path to the SQLite file (opened read-only, per worker).
            event_nos: The events this dataset serves, in order.
            pulsemap: Pulse table to read from.
            sensor_key: Column with the per-pulse sensor id.
        """
        self.db = db
        self.ev = np.asarray(event_nos)
        self.sql = _PULSE_SQL.format(pulsemap=pulsemap, key_col=f",{sensor_key}")
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
        return {
            "event_no": ev,
            "pulses": rows[:, :-1].astype(np.float32),
            "sensor_key": rows[:, -1].astype(np.int64),
        }
