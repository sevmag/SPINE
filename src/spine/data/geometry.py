"""Detector geometry asset: sensor positions + k-NN graph + KDTree.

The CURTAIN sampler needs this (pulse->sensor matching, nearest-dark negatives).
Pure numpy/scipy -- a SPINE asset, not graphnet's. Feature scaling is a separate
concern (spine.data.scaling.FeatureScaler).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def load_geometry(path: str) -> dict:
    """Load a sensor-layout asset and attach a KDTree for pulse matching.

    Args:
        path: An .npz with per-sensor arrays (at least `xyz`; `knn_idx` for
            nearest-dark lookups).

    Returns:
        The asset's arrays plus a cKDTree over `xyz` under the key "tree".
    """
    d = np.load(path)
    geo = {k: d[k] for k in d.files}
    geo["tree"] = cKDTree(geo["xyz"].astype(np.float64))
    return geo
