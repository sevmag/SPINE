"""CURTAIN objectives. v1 = [OCCUPANCY]; v2 = [OCCUPANCY, dt].

Each objective declares how many head channels it consumes and its loss. The
per-objective masking (occupancy over valid queries; Delta-t over hit queries
only) and dt scaling live in `CurtainTask.loss`, which owns the batch layout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from spine.pretext.base import Objective


def _bce(pred: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(pred.squeeze(-1), target.float())


def _smooth_l1(pred: Tensor, target: Tensor) -> Tensor:
    return F.smooth_l1_loss(pred.squeeze(-1), target.float())


OCCUPANCY = Objective(
    name="occupancy", channels=1, target_key="label", loss_fn=_bce, weight=1.0,
)


def dt_objective(weight: float = 1.0) -> Objective:
    """The v2 Delta-t regression objective (charge-weighted-mean-time ref)."""
    return Objective(name="dt", channels=1, target_key="dt",
                     loss_fn=_smooth_l1, weight=weight)
