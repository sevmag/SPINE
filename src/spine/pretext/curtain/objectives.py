"""CURTAIN objectives. v1 = [OCCUPANCY]; v2 = [OCCUPANCY, dt_objective()].

Each objective owns its head and its loss (target + masking): occupancy is BCE
over ALL real queries; Delta-t is SmoothL1 over HIT queries only. That masking
lives here, not in the task -- adding an objective needs no task changes.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from spine.pretext.base import Objective


class OccupancyObjective(Objective):
    """v1: per-query hit-after-T occupancy, BCE over all real queries."""

    name = "occupancy"

    def build_head(self, dim: int) -> nn.Module:
        """One hit logit per query."""
        return nn.Linear(dim, 1)

    def loss(self, pred: Tensor, batch: dict) -> Tensor:
        """BCE over all real queries."""
        return F.binary_cross_entropy_with_logits(
            pred.squeeze(-1), batch["label"].values()
        )


class DtObjective(Objective):
    """v2 add-on: regress cwm-referenced Delta-t on HIT queries only."""

    name = "dt"

    def build_head(self, dim: int) -> nn.Module:
        """One Delta-t regression output per query."""
        return nn.Linear(dim, 1)

    def loss(self, pred: Tensor, batch: dict) -> Tensor:
        """Smooth-L1 over hit queries only."""
        hit = batch["label"].values() > 0.5
        if not hit.any():
            return pred.new_zeros(())
        return F.smooth_l1_loss(pred[hit].squeeze(-1), batch["dt"].values()[hit])


OCCUPANCY = OccupancyObjective()


def dt_objective(weight: float = 1.0) -> Objective:
    """Build the v2 Delta-t regression objective (cwm-referenced target)."""
    return DtObjective(weight=weight)
