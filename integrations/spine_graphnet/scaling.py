"""graphnet Detector -> spine FeatureScaler bridge."""

from __future__ import annotations

from graphnet.models.detector import Detector
from torch import Tensor

from spine.data.scaling import FeatureLayout, FeatureScaler


class DetectorScaler(FeatureScaler):
    """Scale features with a graphnet Detector's feature_map.

    Pretraining in the detector's own standardization keeps the encoder's
    feature space identical to what a downstream graphnet StandardModel
    applies at fine-tune time. Columns without a feature_map entry (e.g. a
    synthetic charge) pass through unscaled; query positions go through the
    detector's xyz entries so they stay on the pulse coordinate scale.
    """

    def __init__(
        self,
        detector: Detector,
        feature_names: list[str],
        layout: FeatureLayout | None = None,
    ):
        """Bind the detector's standardization to the pulse columns.

        Args:
            detector: graphnet Detector supplying feature_map() and xyz.
            feature_names: Name per raw pulse column, in column order.
            layout: Column layout; None uses the default order.
        """
        super().__init__(layout)
        self.detector = detector
        self.feature_names = feature_names

    def scale_pulses(self, x: Tensor) -> Tensor:
        """Apply the detector's per-column standardization.

        Args:
            x: [..., F] raw pulse features, columns per `self.layout`.

        Returns:
            Standardized features, same shape and column order.
        """
        fm = self.detector.feature_map()
        out = x.clone()
        for j, name in enumerate(self.feature_names):
            if name in fm:
                out[..., j] = fm[name](x[..., j])
        return out

    def scale_positions(self, p: Tensor) -> Tensor:
        """Apply the detector's xyz standardization to raw positions.

        Args:
            p: [..., 3] raw positions, same units as the pulse xyz columns.

        Returns:
            Standardized coordinates on the same scale as scaled pulse xyz.
        """
        fm = self.detector.feature_map()
        out = p.clone()
        for k, name in enumerate(self.detector.xyz):
            out[..., k] = fm[name](p[..., k])
        return out
