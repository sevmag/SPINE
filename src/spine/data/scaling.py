"""Feature scaling: raw detector features -> encoder inputs.

Scaling is detector-specific, so it's an abstraction: `FeatureScaler` defines the
contract; a subclass supplies one detector's transforms. It's injected into the
pretext task and applied at the model boundary -- AFTER the pretext split, so the
sampler sees RAW values (CURTAIN's dt reference is charge-weighted-mean-time on
raw pulses; the geometry match is in raw metres).

Two consumers share one scaler: the pulse features (encoder input) and the query
positions (CURTAIN head input), which must use the SAME coordinate scale -- so
both live on the scaler.

Add a geometry by subclassing FeatureScaler; reuse graphnet's scaling by wrapping
its `Detector` in a FeatureScaler (see examples).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class FeatureScaler(ABC):
    """Raw -> standardized feature scaling for one detector.

    Input pulses are [., 5] = (x, y, z, t, charge), raw. `scale_positions` must
    apply the same xyz scaling as the first three channels of `scale_pulses`.
    """

    @abstractmethod
    def scale_pulses(self, x: Tensor) -> Tensor:
        """Raw [., 5] (x, y, z, t, charge) -> standardized encoder features."""
        raise NotImplementedError

    @abstractmethod
    def scale_positions(self, p: Tensor) -> Tensor:
        """Raw [., 3] positions -> standardized coordinates (pretext queries)."""
        raise NotImplementedError


class HexagonScaler(FeatureScaler):
    """NuBench Hexagon scaling. Matches graphnet's Hexagon detector so encoders
    transfer to the supervised DeepIce baseline downstream."""

    _POS = torch.tensor([100.0, 100.0, 1000.0])

    def scale_pulses(self, x: Tensor) -> Tensor:
        out = torch.empty_like(x)
        out[..., :3] = x[..., :3] / self._POS.to(x.device)
        out[..., 3] = x[..., 3] / 1e6
        out[..., 4] = torch.log10(1.0 + x[..., 4].clamp(min=1e-2))
        return out

    def scale_positions(self, p: Tensor) -> Tensor:
        return p / self._POS.to(p.device)
