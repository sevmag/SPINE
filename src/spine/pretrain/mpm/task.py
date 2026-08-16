"""Masked point modeling (MPM): the self-supervised pretext of Yu, Kamp &
Arguelles, "Reducing Simulation Dependence in Neutrino Telescopes with Masked
Point Transformers" (arXiv:2510.01733), reproduced as a SPINE task after the
paper's reference code (the ``prometheus`` branch of github.com/felixyu7/neptune).

Per event, ``max(1, int(mask_ratio * n_hits))`` hits drawn uniformly at random
have their space-time coordinates hidden -- replaced, inside ``NeptuneBackbone``,
by learned mask embeddings -- while their charge stays visible; per-token linear
heads reconstruct the hidden coordinates from the encoder output with a smooth-L1
loss over the masked hits only. ``mode`` selects which coordinates are masked
and scored: "spatial" (xyz), "temporal" (t) or "spatiotemporal" (both), the
paper's ``pretrain_task``. Following the reference, the loss is a per-event mean
over that event's masked hits, then a mean over events, with the xyz and time
terms weighted by ``centroid_loss_weight`` and ``time_loss_weight``.

Reconstruction targets are the hits' original coordinates. xyz is the encoder's
own scaled coordinate (the coordinate the mask replaced), as in the reference,
where target and input share one space. Time cannot share the encoder's input
space here: SPINE's detector scalers put the time input at O(1e-3) (t/1e6, the
downstream graphnet convention), and a smooth-L1 target that small is fitted by
predicting zero. The time target is therefore the referenced pulse time divided
by ``time_scale``; the default (1000 ns) reproduces the reference's unit, which
regresses times in microseconds (raw ns / 1000) referenced to the event's
earliest hit (``time_ref="min"``).

Pairs with ``spine.backbones.neptune.NeptuneBackbone``, whose per-hit token
content is position-free, so a masked hit's coordinate must be inferred from
its charge and the rest of the event. Masking is per hit, the paper's regime
below ``max_tokens`` (every hit its own token). Positions come from the pulses
themselves, so no geometry asset is needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spine.pretrain.base import PretrainTask, Sample

_XYZ = slice(0, 3)
_MODES = ("spatial", "temporal", "spatiotemporal")
_TIME_REFS = ("min", "cwm", None)


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


class MPMTask(PretrainTask):
    """Masked point modeling over per-hit tokens (paper-faithful)."""

    objectives: list = []

    def __init__(
        self,
        geo: dict,
        scaler: Any,
        max_pulses: int = 768,
        time_ref: str | None = "min",
        time_scale: float = 1000.0,
        mask_ratio: float = 0.15,
        mode: str = "spatial",
        centroid_loss_weight: float = 1.0,
        time_loss_weight: float = 1.0,
    ):
        """Assemble the MPM task.

        The defaults for ``mask_ratio`` and ``mode`` are those of the reference
        code; the paper's directional-reconstruction experiment used
        ``mode="temporal", mask_ratio=1.0`` (see configs/task/mpm.yaml).

        Args:
            geo: Geometry asset (unused; positions come from the pulses).
            scaler: Detector feature scaling, applied at collate time.
            max_pulses: Cap on hits fed to the encoder per event (random
                subsample of the event).
            time_ref: Per-event time reference subtracted from every pulse
                time before scaling: "min" (earliest hit, the reference
                code's ``t - t.min()``), "cwm" (charge-weighted mean, the
                CURTAIN convention) or None (raw times).
            time_scale: Divisor (ns per unit) mapping the referenced time to
                the reconstruction-target unit; 1000 = microseconds.
            mask_ratio: Fraction of hits masked per event; the count is
                ``max(1, int(mask_ratio * n))``, so 1.0 masks every hit.
            mode: Which coordinates to mask and score -- "spatial",
                "temporal" or "spatiotemporal".
            centroid_loss_weight: Weight on the xyz reconstruction term.
            time_loss_weight: Weight on the time reconstruction term.

        Raises:
            ValueError: If mode or time_ref is not a recognized option.
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if time_ref not in _TIME_REFS:
            raise ValueError(f"time_ref must be one of {_TIME_REFS}, got {time_ref!r}")
        self.geo = geo
        self.scaler = scaler
        self.max_pulses = max_pulses
        self.time_ref = time_ref
        self.time_scale = time_scale
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.centroid_loss_weight = centroid_loss_weight
        self.time_loss_weight = time_loss_weight

    def make_sample(self, event: dict[str, np.ndarray], rng: np.random.Generator) -> Sample:
        """Reference times, cap the hits, and draw this event's masked subset."""
        p = np.asarray(event["pulses"], dtype=np.float32)
        if len(p) == 0:
            raise ValueError(f"event {event['event_no']} has no pulses")
        lay = self.scaler.layout
        # The reference is taken over the whole event, before the cap, so a
        # subsampled event keeps the same time origin as the full one.
        if self.time_ref == "min":
            t0 = float(p[:, lay.t].min())
        elif self.time_ref == "cwm":
            w = np.clip(p[:, lay.charge], 0.0, None) + 1e-6
            t0 = float((w * p[:, lay.t]).sum() / w.sum())
        else:
            t0 = 0.0
        if len(p) > self.max_pulses:
            p = p[rng.choice(len(p), self.max_pulses, replace=False)]
        p = p.copy()
        p[:, lay.t] -= t0
        n = len(p)
        n_mask = max(1, int(self.mask_ratio * n))
        pos_mask = np.zeros(n, dtype=bool)
        pos_mask[rng.permutation(n)[:n_mask]] = True
        return dict(
            pulses=p,
            t_target=(p[:, lay.t] / self.time_scale).astype(np.float32),
            pos_mask=pos_mask,
        )

    def collate(self, samples: list[Sample]) -> dict:
        """Standardize per event; pack pulses jagged, mask and time target padded."""

        def jag(tensors):
            return torch.nested.nested_tensor(tensors, layout=torch.jagged)

        def pad(arrays, fill):
            out = torch.full((len(arrays), lmax), fill, dtype=arrays[0].dtype)
            for b, a in enumerate(arrays):
                out[b, : len(a)] = a
            return out

        scaled = [self.scaler.scale_pulses(torch.from_numpy(s["pulses"])) for s in samples]
        lmax = max(t.shape[0] for t in scaled)
        # qpos/label satisfy the engine's collate contract: qpos is unused by
        # MPMHead; label sizes the logged batch as the number of events, the
        # unit the per-event loss reduction averages over.
        return dict(
            pulses=jag(scaled),
            qpos=jag([t[:, _XYZ].clone() for t in scaled]),
            label=jag([torch.ones(1) for _ in samples]),
            pos_mask=pad([torch.from_numpy(s["pos_mask"]) for s in samples], False),
            t_target=pad([torch.from_numpy(s["t_target"]) for s in samples], 0.0),
            pos_mask_mode=self.mode,
        )

    def build_head(self, dim: int) -> nn.Module:
        """Construct the per-hit reconstruction head."""
        return MPMHead(dim)

    def loss(self, output: Any, batch: dict) -> tuple[Tensor, dict[str, float]]:
        """Smooth-L1 reconstruction of masked hits' coordinates.

        Each term is a mean over the event's masked hits (and channels), then
        a mean over the events of the batch, as in the reference code.
        """
        pred_xyz, pred_t = output
        x0 = batch["pulses"].to_padded_tensor(0.0)
        lengths = batch["pulses"].offsets().diff()
        valid = torch.arange(x0.shape[1], device=x0.device)[None] < lengths[:, None]
        m = batch["pos_mask"].to(valid.device) & valid
        n_masked = m.sum(1)
        scored = n_masked > 0

        def per_event_mean(elem: Tensor) -> Tensor:
            per_hit = elem.mean(-1) * m
            per_event = per_hit.sum(1) / n_masked.clamp(min=1)
            return per_event[scored].mean() if scored.any() else elem.new_zeros(())

        total = pred_xyz.new_zeros(())
        metrics: dict[str, float] = {}
        if self.mode in ("spatial", "spatiotemporal"):
            term = per_event_mean(
                F.smooth_l1_loss(pred_xyz, x0[..., _XYZ], reduction="none")
            )
            total = total + self.centroid_loss_weight * term
            metrics["loss_mpm_centroid"] = float(term.detach())
        if self.mode in ("temporal", "spatiotemporal"):
            t_target = batch["t_target"].to(pred_t.device).unsqueeze(-1)
            term = per_event_mean(F.smooth_l1_loss(pred_t, t_target, reduction="none"))
            total = total + self.time_loss_weight * term
            metrics["loss_mpm_time"] = float(term.detach())
        return total, metrics
