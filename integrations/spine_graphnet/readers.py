"""Adapters exposing graphnet Datasets as SPINE read Datasets."""

from __future__ import annotations

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
