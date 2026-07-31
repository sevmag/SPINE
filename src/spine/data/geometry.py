"""Detector geometry asset: sensor positions + k-NN graph + KDTree.

Canonical loader for the geometry npz (xyz [S,3], knn_idx [S,k]). The CURTAIN
sampler currently carries its own copy for self-containment; unify here once the
data layer owns geometry (see DESIGN).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


def load_geometry(path: str) -> Dict:
    d = np.load(path)
    geo = {k: d[k] for k in d.files}
    geo["tree"] = cKDTree(geo["xyz"].astype(np.float64))
    return geo
