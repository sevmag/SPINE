"""The detector: geometry (sensor layout) + feature standardization.

Two facets of one concept -- what the detector *is* (sensor positions, k-NN
graph, and a KDTree for pulse->sensor matching) and how its raw features map to
encoder inputs (per-feature scaling matching graphnet's NuBench Hexagon). This
is what graphnet's `Detector` bundles too.

Standardization runs at the model boundary, AFTER the pretext split, so the
sampler sees RAW values (CURTAIN's dt reference is charge-weighted-mean-time on
raw pulses; the geometry match is in raw metres). If the backbone reads through
a graphnet GraphDefinition, prefer that Detector over this hand copy.

For multiple geometries (hexagon, flower, ...) this is the natural place to grow
a `Detector` class parametrized by the geometry asset (DESIGN: multi-detector).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

# Per-axis position scale + the time/charge transforms below match graphnet's
# NuBench Hexagon detector, so encoders here see the same feature scales as the
# supervised DeepIce baseline downstream.
_POS = torch.tensor([100.0, 100.0, 1000.0])


def load_geometry(path: str) -> Dict:
    """Sensor-layout asset (xyz, knn_idx, ...) + a KDTree for pulse matching."""
    d = np.load(path)
    geo = {k: d[k] for k in d.files}
    geo["tree"] = cKDTree(geo["xyz"].astype(np.float64))
    return geo


def standardize_pulses(x: Tensor) -> Tensor:
    """Raw [., 5] (x, y, z, t, charge) -> standardized encoder features."""
    out = torch.empty_like(x)
    out[..., :3] = x[..., :3] / _POS.to(x.device)
    out[..., 3] = x[..., 3] / 1e6
    out[..., 4] = torch.log10(1.0 + x[..., 4].clamp(min=1e-2))
    return out


def standardize_pos(p: Tensor) -> Tensor:
    """Raw [., 3] sensor positions -> standardized coordinates."""
    return p / _POS.to(p.device)
