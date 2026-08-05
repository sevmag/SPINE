"""Adapters exposing graphnet Datasets as SPINE read Datasets."""

from __future__ import annotations

import pickle

import numpy as np
from torch.utils.data import Dataset


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
        reader = GraphNetRawDataset(gn, sensor_key_index=...)
    """

    def __init__(
        self,
        gn_dataset: Dataset,
        sensor_key_index: int,
        event_no_key: str = "event_no",
    ):
        """Wrap a graphnet Dataset.

        Args:
            gn_dataset: The graphnet Dataset to adapt (raw features, no
                truth); include the sensor-id column among its features.
            sensor_key_index: Which feature column holds the sensor id; it is
                split out of the pulses into `sensor_key`.
            event_no_key: Attribute on each Data carrying the event id.
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
        j = self.sensor_key_index
        feat = np.delete(x, j, axis=1).astype(np.float32)
        return {
            "event_no": ev,
            "pulses": feat,
            "sensor_key": x[:, j].astype(np.int64),
        }


class LmdbRawDataset(Dataset):
    """Read a GraphNeT LMDBWriter database (pickle) into the read contract.

    Each LMDB value is a pickled dict {pulsemap: {feature: list}, ...}; the key
    is bytes(str(event_no)). Emits pulses in (dom_x, dom_y, dom_z, dom_time,
    charge) order and the per-pulse sensor id `string*100 + dom_number` (IceCube
    DOMs are single-PMT, so no PMT level). Match this key to the geometry
    asset's `dom_key` array.

    The environment is opened lazily per worker: a live LMDB handle cannot cross
    the DataLoader worker start (dropped in __getstate__).
    """

    def __init__(
        self,
        lmdb_path: str,
        event_nos,
        pulsemap: str = "SRTInIcePulses",
    ):
        """Bind the reader to an LMDB and an event list.

        Args:
            lmdb_path: Path to the merged .lmdb directory (opened read-only).
            event_nos: The events this dataset serves, in order.
            pulsemap: Pulse table (top-level dict key) to read from.
        """
        self.path = lmdb_path
        self.ev = np.asarray(event_nos)
        self.pulsemap = pulsemap
        self._env = None

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_env"] = None  # a live LMDB env can't cross the worker start
        return s

    def _begin(self):
        if self._env is None:
            # lmdb is an optional dependency; only this reader needs it, so it is
            # imported here rather than at module import of spine_graphnet.
            import lmdb

            self._env = lmdb.open(
                self.path, readonly=True, lock=False, subdir=True,
                readahead=False,
            )
        return self._env.begin(write=False)

    def __len__(self) -> int:
        return len(self.ev)

    def __getitem__(self, idx: int):
        ev = int(self.ev[idx])
        with self._begin() as txn:
            d = pickle.loads(txn.get(str(ev).encode()))
        pm = d[self.pulsemap]
        pulses = np.column_stack(
            [
                np.asarray(pm["dom_x"], np.float32),
                np.asarray(pm["dom_y"], np.float32),
                np.asarray(pm["dom_z"], np.float32),
                np.asarray(pm["dom_time"], np.float32),
                np.asarray(pm["charge"], np.float32),
            ]
        )
        sensor_key = (
            np.asarray(pm["string"], np.int64) * 100
            + np.asarray(pm["dom_number"], np.int64)
        )
        return {"event_no": ev, "pulses": pulses, "sensor_key": sensor_key}
