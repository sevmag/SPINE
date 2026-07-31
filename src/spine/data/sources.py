"""Event sources -- swap the read layer without touching the pretext.

An `EventSource.read(event_no)` returns one raw event: {"event_no", "pulses"}
where pulses is [P,5] raw (x, y, z, t, charge). The pretext sampler is
source-agnostic (it takes five raw arrays), so SQLite vs LMDB is localized here.

`SqliteSource` is the current path (per-worker lazy connection, point-query).
`LmdbSource` is the fast alternative (see DESIGN): wrap graphnet's lmdb read
utilities behind this interface -- returning RAW pulses (identity detector,
NodesAsPulses, no truth/labels) so standardization still happens after the
pretext split. Left as a stub until the loader is profiled data-bound.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

_PULSE_SQL = (
    "SELECT sensor_pos_x,sensor_pos_y,sensor_pos_z,t,charge "
    "FROM merged_photons WHERE event_no=?"
)


class EventSource(ABC):
    @abstractmethod
    def read(self, event_no: int) -> Dict:
        """-> {'event_no': int, 'pulses': np.ndarray[P,5]}."""
        raise NotImplementedError


class SqliteSource(EventSource):
    """Per-worker lazy SQLite connection; fork-safe (drops handle on pickle)."""

    def __init__(self, db: str, pulsemap: str = "merged_photons"):
        self.db = db
        self.sql = _PULSE_SQL.replace("merged_photons", pulsemap)
        self._con = None

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_con"] = None  # a live handle can't cross the fork
        return s

    def _cur(self):
        if self._con is None:
            self._con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        return self._con.cursor()

    def read(self, event_no: int) -> Dict:
        rows = self._cur().execute(self.sql, (int(event_no),)).fetchall()
        return {"event_no": int(event_no),
                "pulses": np.asarray(rows, np.float32)}


class LmdbSource(EventSource):
    """Fast LMDB read (raw pulses). TODO: wrap graphnet lmdb_utilities."""

    def __init__(self, path: str, pulsemap: str = "merged_photons"):
        self.path = path
        self.pulsemap = pulsemap

    def read(self, event_no: int) -> Dict:
        raise NotImplementedError(
            "LmdbSource: wire graphnet.data.utilities.lmdb_utilities "
            "(get_all_indices / get_serialization_method) to return raw pulses."
        )
