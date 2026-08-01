"""Detector geometry asset: sensor positions + k-NN graph + KDTree.

The CURTAIN sampler needs this (pulse->sensor matching, nearest-dark negatives).
Pure numpy/scipy -- a SPINE asset, not graphnet's. Feature scaling is a separate
concern (spine.data.scaling.FeatureScaler).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def load_geometry(path: str, key: str | None = None) -> dict:
    """Load a sensor-layout asset and attach the pulse-matching structures.

    Args:
        path: An .npz with per-sensor arrays (at least `xyz`; `knn_idx` for
            nearest-dark lookups; optionally per-row sensor-key arrays).
        key: Name of a stored per-row array of unique integer sensor keys
            (e.g. "pmt_id"). When given, a key -> row lookup is built so
            readers can identify sensors by ID instead of coordinates.

    Returns:
        The asset's arrays plus a cKDTree over `xyz` under "tree" and, when
        `key` is given, the lookup dict under "key_to_row".

    Raises:
        ValueError: If `key` names no stored array or the keys are not
            unique per row.
    """
    d = np.load(path)
    geo = {k: d[k] for k in d.files}
    geo["tree"] = cKDTree(geo["xyz"].astype(np.float64))
    if key is not None:
        if key not in geo:
            raise ValueError(f"geometry asset has no per-row array {key!r}")
        keys = geo[key].astype(np.int64)
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"sensor keys in {key!r} are not unique per row")
        geo["key_to_row"] = {int(k): i for i, k in enumerate(keys)}
    return geo
