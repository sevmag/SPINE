"""Feature scaling: raw detector features -> encoder inputs.

Scaling is detector-specific, so it's an abstraction: `FeatureScaler` defines the
contract; a subclass supplies one detector's transforms. It's injected into the
pretext task and applied at the model boundary -- AFTER the pretext split, so the
sampler sees RAW values (CURTAIN's dt reference is charge-weighted-mean-time on
raw pulses; the geometry match is in raw metres).

Column order is NOT hardcoded: a `FeatureLayout` names which column is x/y/z/t/
charge (default matches the reader contract), so a reader emitting a different
order just needs a different layout. The scaler holds it so the sampler and the
scaler share one source of truth.

Two consumers share one scaler: pulse features (encoder input) and query
positions (CURTAIN head input), which must use the SAME coordinate scale.
Add a geometry by subclassing FeatureScaler; wrap graphnet's `Detector` in one
to reuse its scaling (see examples).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureLayout:
    """Which column of the raw pulse array holds each feature.

    Default matches the RawPulseDataset contract (x, y, z, t, charge). Override
    if your reader emits a different order -- nothing indexes by magic number.
    """

    x: int = 0
    y: int = 1
    z: int = 2
    t: int = 3
    charge: int = 4

    @property
    def pos(self) -> tuple[int, int, int]:
        """The (x, y, z) column indices."""
        return (self.x, self.y, self.z)


class FeatureScaler(ABC):
    """Raw -> standardized feature scaling for one detector.

    `scale_positions` must apply the same xyz scaling as the position columns of
    `scale_pulses`; both index features through `self.layout`.
    """

    def __init__(self, layout: FeatureLayout | None = None):
        self.layout = layout or FeatureLayout()

    @abstractmethod
    def scale_pulses(self, x: Tensor) -> Tensor:
        """Raw pulses -> standardized encoder features (columns per layout)."""
        raise NotImplementedError

    @abstractmethod
    def scale_positions(self, p: Tensor) -> Tensor:
        """Raw [., 3] positions -> standardized coordinates (pretext queries)."""
        raise NotImplementedError


class HexagonScaler(FeatureScaler):
    """NuBench Hexagon feature scaling.

    Matches graphnet's Hexagon detector so encoders transfer to the supervised
    DeepIce baseline downstream.
    """

    _POS = torch.tensor([100.0, 100.0, 1000.0])

    def scale_pulses(self, x: Tensor) -> Tensor:
        """Scale xyz and t to detector units; log-compress the charge."""
        lay = self.layout
        out = x.clone()
        out[..., list(lay.pos)] = x[..., list(lay.pos)] / self._POS.to(x.device)
        out[..., lay.t] = x[..., lay.t] / 1e6
        out[..., lay.charge] = torch.log10(1.0 + x[..., lay.charge].clamp(min=1e-2))
        return out

    def scale_positions(self, p: Tensor) -> Tensor:
        """Scale raw positions with the same xyz factors as the pulses."""
        return p / self._POS.to(p.device)
