"""Masked point modeling (MPM): the self-supervised pretext of Yu, Kamp &
Arguelles, "Reducing Simulation Dependence in Neutrino Telescopes with Masked
Point Transformers" (arXiv:2510.01733), reproduced as a SPINE task.

A random fraction of hits have their space-time coordinates hidden -- replaced,
inside ``NeptuneBackbone``, by learned mask embeddings -- while their charge
stays visible; the head reconstructs the hidden coordinates from the encoder's
per-hit outputs with a smooth-L1 loss. ``mode`` selects which coordinates are
masked and scored: "spatial" (xyz), "temporal" (t) or "spatiotemporal" (both),
matching the paper's ``pretrain_task``.

Pairs with ``spine.backbones.neptune.NeptuneBackbone``, whose per-hit token
content is position-free, so recovering a masked hit's coordinate is non-trivial
(it must be inferred from the hit's charge and the rest of the event). The mask
is per-hit, faithful to the paper's below-``max_tokens`` regime. Positions are
reconstructed from the pulses themselves, so no geometry asset is needed.

Coordinate columns follow the ``NeptuneBackbone`` default position layout
``(x, y, z, t)`` at columns 0-3 of the scaled pulse feature vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spine.pretext.base import PretextTask, Sample

_XYZ = slice(0, 3)
_T = slice(3, 4)


class MPMHead(nn.Module):
    """Per-hit coordinate reconstruction heads (centroid xyz + time)."""

    def __init__(self, dim: int):
        """Build the xyz and time linear predictors over encoder tokens."""
        super().__init__()
        self.centroid = nn.Linear(dim, 3)
        self.time = nn.Linear(dim, 1)

    def forward(self, _query_pos: Tensor, enc: Any) -> tuple[Tensor, Tensor]:
        """Predict per-hit xyz and time from the encoded tokens.

        The query-position argument the engine passes is unused: MPM
        reconstructs one prediction per encoder token, not per query.
        """
        return self.centroid(enc.tokens), self.time(enc.tokens)


class MPMTask(PretextTask):
    """Masked point modeling over per-hit tokens (paper-faithful)."""

    objectives: list = []

    def __init__(
        self,
        geo: dict,
        scaler: Any,
        max_pulses: int = 768,
        center_time: bool = True,
        mask_ratio: float = 0.15,
        mode: str = "temporal",
        centroid_loss_weight: float = 1.0,
        time_loss_weight: float = 1.0,
    ):
        """Assemble the MPM task.

        Args:
            geo: Geometry asset (unused; positions come from the pulses).
            scaler: Detector feature scaling, applied at collate time.
            max_pulses: Cap on hits fed to the encoder per event.
            center_time: Reference times to the charge-weighted mean.
            mask_ratio: Fraction of hits masked per event (1.0 masks all,
                the paper's directional setting).
            mode: Which coordinates to mask and score -- "spatial",
                "temporal" or "spatiotemporal".
            centroid_loss_weight: Weight on the xyz reconstruction term.
            time_loss_weight: Weight on the time reconstruction term.

        Raises:
            ValueError: If mode is not one of the three tasks.
        """
        if mode not in ("spatial", "temporal", "spatiotemporal"):
            raise ValueError(
                f"mode must be spatial|temporal|spatiotemporal, got {mode!r}"
            )
        self.geo = geo
        self.scaler = scaler
        self.max_pulses = max_pulses
        self.center_time = center_time
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.centroid_loss_weight = centroid_loss_weight
        self.time_loss_weight = time_loss_weight

    def make_sample(self, event: dict[str, np.ndarray], rng: np.random.Generator) -> Sample:
        """Cap and time-center one event's hits (no split -- MPM masks in-place)."""
        p = event["pulses"]
        lay = self.scaler.layout
        if len(p) > self.max_pulses:
            p = p[rng.choice(len(p), self.max_pulses, replace=False)]
        p = p.astype(np.float32).copy()
        if self.center_time:
            w = np.clip(p[:, lay.charge], 0.0, None) + 1e-6
            p[:, lay.t] -= float((w * p[:, lay.t]).sum() / w.sum())
        return dict(pulses=p)

    def collate(self, samples: list[Sample]) -> dict:
        """Pack pulses and draw a per-event random position mask."""

        def jag(tensors):
            return torch.nested.nested_tensor(tensors, layout=torch.jagged)

        scaled = [self.scaler.scale_pulses(torch.from_numpy(s["pulses"])) for s in samples]
        lengths = [t.shape[0] for t in scaled]
        lmax = max(lengths)
        pos_mask = torch.zeros(len(samples), lmax, dtype=torch.bool)
        for b, n in enumerate(lengths):
            k = max(1, int(round(self.mask_ratio * n)))
            pos_mask[b, torch.randperm(n)[:k]] = True
        # qpos/label satisfy the engine collate contract; qpos is unused by
        # MPMHead and label only sizes the logged batch (total hits).
        return dict(
            pulses=jag(scaled),
            qpos=jag([t[:, _XYZ].clone() for t in scaled]),
            label=jag([torch.ones(n) for n in lengths]),
            pos_mask=pos_mask,
            pos_mask_mode=self.mode,
        )

    def build_head(self, dim: int) -> nn.Module:
        """Construct the per-hit reconstruction head."""
        return MPMHead(dim)

    def loss(self, output: Any, batch: dict) -> tuple[Tensor, dict[str, float]]:
        """Smooth-L1 reconstruction of masked hits' coordinates."""
        pred_xyz, pred_t = output
        x0 = batch["pulses"].to_padded_tensor(0.0)
        lengths = batch["pulses"].offsets().diff()
        valid = torch.arange(x0.shape[1], device=x0.device)[None] < lengths[:, None]
        m = batch["pos_mask"].to(valid.device) & valid
        total = pred_xyz.new_zeros(())
        metrics: dict[str, float] = {}
        if self.mode in ("spatial", "spatiotemporal") and m.any():
            term = F.smooth_l1_loss(pred_xyz[m], x0[..., _XYZ][m])
            total = total + self.centroid_loss_weight * term
            metrics["loss_mpm_centroid"] = float(term.detach())
        if self.mode in ("temporal", "spatiotemporal") and m.any():
            term = F.smooth_l1_loss(pred_t[m], x0[..., _T][m])
            total = total + self.time_loss_weight * term
            metrics["loss_mpm_time"] = float(term.detach())
        return total, metrics
