"""CURTAIN objectives. v1 = [OccupancyObjective]; v2 adds DtObjective.

Each objective owns its head and loss, including its masking -- occupancy
scores all real queries, dt only the hit ones.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from spine.pretext.base import Objective


class OccupancyObjective(Objective):
    """v1: per-query hit-after-T occupancy, BCE over all real queries."""

    name = "occupancy"

    def build_head(self, dim: int) -> nn.Module:
        """Build the occupancy head.

        Args:
            dim: Width of the shared per-query embedding.

        Returns:
            Linear head emitting one hit logit per query.
        """
        return nn.Linear(dim, 1)

    def loss(self, pred: Tensor, batch: dict) -> Tensor:
        """BCE over all real queries.

        Args:
            pred: [sum_Q, 1] hit logits.
            batch: Collated batch; targets under "label".

        Returns:
            Scalar BCE loss.
        """
        return F.binary_cross_entropy_with_logits(
            pred.squeeze(-1), batch["label"].values()
        )


class DtObjective(Objective):
    """v2 add-on: regress cwm-referenced Delta-t on HIT queries only."""

    name = "dt"

    def build_head(self, dim: int) -> nn.Module:
        """Build the Delta-t head.

        Args:
            dim: Width of the shared per-query embedding.

        Returns:
            Linear head emitting one Delta-t value per query.
        """
        return nn.Linear(dim, 1)

    def loss(self, pred: Tensor, batch: dict) -> Tensor:
        """Smooth-L1 over hit queries only.

        Args:
            pred: [sum_Q, 1] Delta-t predictions.
            batch: Collated batch; targets under "dt", hit mask from "label".

        Returns:
            Scalar loss (zero when the batch has no hit queries).
        """
        hit = batch["label"].values() > 0.5
        if not hit.any():
            return pred.new_zeros(())
        return F.smooth_l1_loss(pred[hit].squeeze(-1), batch["dt"].values()[hit])
