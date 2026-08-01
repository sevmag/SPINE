"""Detector geometry asset: sensor positions + k-NN graph + KDTree.

The CURTAIN sampler needs this (pulse->sensor matching, nearest-dark negatives).
Pure numpy/scipy -- a SPINE asset, not graphnet's. Feature scaling is a separate
concern (spine.data.scaling.FeatureScaler).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def load_geometry(path: str) -> dict:
    """Sensor-layout asset (xyz, knn_idx, ...) + a KDTree for pulse matching."""
    d = np.load(path)
    geo = {k: d[k] for k in d.files}
    geo["tree"] = cKDTree(geo["xyz"].astype(np.float64))
    return geo
