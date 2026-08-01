"""Detector geometry asset: sensor positions, k-NN graph, sensor keys.

The CURTAIN sampler needs this (query positions, nearest-dark negatives) and
sensor identity comes from it (data-carried keys -> geometry rows). Pure
numpy -- a SPINE asset, not graphnet's. Feature scaling is a separate concern
(spine.data.scaling.FeatureScaler).
"""

from __future__ import annotations

import numpy as np


def load_geometry(path: str, sensor_key: str | None = None) -> dict:
    """Load a sensor-layout asset and attach the pulse-matching structures.

    Args:
        path: An .npz with per-sensor arrays (at least `xyz`; `knn_idx` for
            nearest-dark lookups; optionally per-row sensor-key arrays).
        sensor_key: Name of a stored per-row array of unique integer
            sensor keys (e.g. "pmt_id"). When given, a key -> row lookup is
            built so readers can identify sensors by ID instead of
            coordinates.

    Returns:
        The asset's arrays plus, when `sensor_key` is given, the lookup dict
        under "sensor_key_to_row".

    Raises:
        ValueError: If `sensor_key` names no stored array or the keys are
            not unique per row.
    """
    d = np.load(path)
    geo = {k: d[k] for k in d.files}
    if sensor_key is not None:
        if sensor_key not in geo:
            raise ValueError(f"geometry asset has no per-row array {sensor_key!r}")
        keys = geo[sensor_key].astype(np.int64)
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"sensor keys in {sensor_key!r} are not unique per row")
        geo["sensor_key_to_row"] = {int(k): i for i, k in enumerate(keys)}
    return geo
