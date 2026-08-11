"""CURTAIN objectives. v1 = [OccupancyObjective]; v2 adds DtObjective.

Each objective owns its head and loss, including its masking -- occupancy
scores all real queries, dt only the hit ones.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spine.pretrain.base import Objective


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
        """BCE over all real queries, as a per-event-weighted mean.

        Args:
            pred: [sum_Q, 1] hit logits.
            batch: Collated batch; targets under "label", per-query event
                weights under "w" (all 1.0 without a weight table, where
                this reduces exactly to the unweighted mean).

        Returns:
            Scalar BCE loss.
        """
        per_query = F.binary_cross_entropy_with_logits(
            pred.squeeze(-1), batch["label"].values(), reduction="none"
        )
        w = batch["w"].values()
        return (w * per_query).sum() / w.sum().clamp_min(1e-8)


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
        per_query = F.smooth_l1_loss(
            pred[hit].squeeze(-1), batch["dt"].values()[hit], reduction="none"
        )
        w = batch["w"].values()[hit]
        return (w * per_query).sum() / w.sum().clamp_min(1e-8)


class ChargeObjective(Objective):
    """v3 add-on: probabilistic total-charge on HIT queries.

    Gaussian NLL of log10(1+Q_total): the head emits (mu, log_var) and the
    learned variance absorbs the irreducible photon/PMT noise instead of
    spending trunk capacity fitting it. Charge is the light-yield signal that
    energy reconstruction most directly needs.
    """

    name = "charge"
    _LOGVAR_CLAMP = 7.0

    def build_head(self, dim: int) -> nn.Module:
        """Two channels per query: mean and log-variance of log10(1+Q)."""
        return nn.Linear(dim, 2)

    def loss(self, pred: Tensor, batch: dict) -> Tensor:
        """Gaussian NLL over hit queries only (zero when the batch has none)."""
        hit = batch["label"].values() > 0.5
        if not hit.any():
            return pred.new_zeros(())
        mu = pred[hit][:, 0]
        s = pred[hit][:, 1].clamp(-self._LOGVAR_CLAMP, self._LOGVAR_CLAMP)
        err = mu - batch["q"].values()[hit]
        nll = 0.5 * (torch.exp(-s) * err.pow(2) + s)
        w = batch["w"].values()[hit]
        return (w * nll).sum() / w.sum().clamp_min(1e-8)
