"""Detector feature scaling, applied at the model boundary.

The sampler sees RAW values (its dt reference and geometry live in raw
units); collate standardizes. `FeatureLayout` names the raw columns so
nothing indexes by magic number; pulse features and query positions share
one scaler so coordinates stay on one scale.
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
        """Store the column layout shared by all scaling calls.

        Args:
            layout: Which raw column holds which feature; None uses the
                default (x, y, z, t, charge) order.
        """
        self.layout = layout or FeatureLayout()

    @abstractmethod
    def scale_pulses(self, x: Tensor) -> Tensor:
        """Standardize raw pulse features for the encoder.

        Args:
            x: [..., F] raw pulse features, columns per `self.layout`.

        Returns:
            Standardized features, same shape and column order.
        """
        ...

    @abstractmethod
    def scale_positions(self, p: Tensor) -> Tensor:
        """Standardize raw positions (pretext query coordinates).

        Args:
            p: [..., 3] raw positions, same units as the pulse xyz columns.

        Returns:
            Standardized coordinates on the same scale as scaled pulse xyz.
        """
        ...


class HexagonScaler(FeatureScaler):
    """NuBench Hexagon feature scaling.

    Matches graphnet's Hexagon detector so encoders transfer to the supervised
    DeepIce baseline downstream.
    """

    _POS = torch.tensor([100.0, 100.0, 1000.0])

    def scale_pulses(self, x: Tensor) -> Tensor:
        """Scale xyz and t to detector units; log-compress the charge.

        Args:
            x: [..., F] raw pulse features, columns per `self.layout`.

        Returns:
            Standardized features, same shape and column order.
        """
        lay = self.layout
        out = x.clone()
        out[..., list(lay.pos)] = x[..., list(lay.pos)] / self._POS.to(x.device)
        out[..., lay.t] = x[..., lay.t] / 1e6
        out[..., lay.charge] = torch.log10(1.0 + x[..., lay.charge].clamp(min=1e-2))
        return out

    def scale_positions(self, p: Tensor) -> Tensor:
        """Scale raw positions with the same xyz factors as the pulses.

        Args:
            p: [..., 3] raw positions in metres.

        Returns:
            Standardized coordinates on the same scale as scaled pulse xyz.
        """
        return p / self._POS.to(p.device)
