"""CurtainTask -- assembles sampler + head + objectives into a PretextTask.

make_sample runs the CURTAIN sampler on RAW pulses (fresh RNG per call = a new
split each epoch); collate standardizes each event and packs the batch as jagged
nested tensors (pulses + query fields) with no padding baked in -- the backbone
and head project via to_padded_tensor; build_head assembles the shared query
encoder with one head per objective; loss
sums each objective's own loss (which owns its target lookup + masking)
over the real queries, weighted.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from spine.data.scaling import FeatureScaler
from spine.pretext.base import Objective, PretextTask, Sample
from spine.pretext.curtain.head import MultiObjectiveHead
from spine.pretext.curtain.sampler import sample_event


def real_query_mask(pred: Tensor, batch: dict) -> Tensor:
    """Mask the real (non-padded) queries of a padded prediction.

    Real queries pack at the front of each row (to_padded_tensor), so the
    per-event query counts from the qpos NJT select them.

    Args:
        pred: [B, Qmax, ...] padded per-query predictions.
        batch: Collated batch; query lengths come from the qpos NJT.

    Returns:
        [B, Qmax] bool mask, True at real queries.
    """
    qlen = batch["qpos"].offsets().diff()
    return torch.arange(pred.shape[1], device=pred.device)[None] < qlen[:, None]


class CurtainTask(PretextTask):
    """Occupancy(/+dt) forecast over held-out sensors of a split event."""

    def __init__(
        self,
        geo: dict,
        objectives: list[Objective],
        scaler: FeatureScaler,
        max_pulses: int = 768,
        center_time: bool = True,
        dt_scale: float = 500.0,
        holdout_mode: str = "temporal",
        random_vis_frac: float = 0.5,
        q_lo: float = 0.3,
        q_hi: float = 0.7,
        pos_k: int = 32,
        neg_anchor: str = "hidden",
        rand_neg_frac: float = 0.15,
        min_visible: int = 8,
        min_future: int = 4,
        resample_tries: int = 6,
    ):
        """Assemble the task from its parts.

        Args:
            geo: Geometry asset (spine.data.geometry.load_geometry).
            objectives: Scored objectives; also sizes the head.
            scaler: Detector feature scaling, applied at collate time.
            max_pulses: Cap on visible pulses fed to the encoder per event.
            center_time: Reference pulse times to the charge-weighted mean
                time of the visible pulses.
            dt_scale: Divisor bringing the dt regression target to O(1).
            holdout_mode: "temporal" (time cutoff) or "random" (sensor split).
            random_vis_frac: Visible fraction of the hit sensors; only
                consulted in "random" mode.
            q_lo: Lower bound of the cutoff-quantile window.
            q_hi: Upper bound of the cutoff-quantile window.
            pos_k: Maximum positive queries per event (capped by supply).
            neg_anchor: "hidden" (nearest dark to each positive) or
                "visible-front" (nearest dark to the latest visible sensors).
            rand_neg_frac: Fraction of negatives drawn fully at random.
            min_visible: Minimum visible sensors for a valid split.
            min_future: Minimum future-new sensors for a valid split.
            resample_tries: Random cutoffs before the deterministic fallback.
        """
        self.geo = geo
        self.objectives = objectives
        self.scaler = scaler
        self.max_pulses = max_pulses
        self.center_time = center_time
        self.dt_scale = dt_scale
        self.holdout_mode = holdout_mode
        self.random_vis_frac = random_vis_frac
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.pos_k = pos_k
        self.neg_anchor = neg_anchor
        self.rand_neg_frac = rand_neg_frac
        self.min_visible = min_visible
        self.min_future = min_future
        self.resample_tries = resample_tries

    # ---- data side (CPU, per event / per batch) -------------------------
    def make_sample(
        self, event: dict[str, np.ndarray], rng: np.random.Generator
    ) -> Sample:
        """Split one event's raw pulses into encoder input + query targets.

        Args:
            event: One raw event from the read layer.
            rng: Per-call generator (fresh entropy = new split per epoch).

        Returns:
            The per-event sample: visible pulses, query positions and targets.

        Raises:
            ValueError: If the event cannot be split or leaves too few
                visible pulses; pre-filter the selection to usable events.
        """
        p = event["pulses"]  # [P, n] raw; columns per self.scaler.layout
        lay = self.scaler.layout
        keys = event.get("sensor_key")
        if keys is None:
            raise ValueError(
                f"event {event['event_no']} carries no sensor_key -- the "
                "reader must emit per-pulse sensor identity (see the "
                "RawPulseDataset contract)"
            )
        lookup = self.geo.get("sensor_key_to_row")
        if lookup is None:
            raise ValueError(
                "reader provides sensor_key but the geometry was loaded "
                "without one -- pass sensor_key=... to load_geometry"
            )
        try:
            sensor = np.fromiter(
                (lookup[int(k)] for k in keys), np.int64, count=len(keys)
            )
        except KeyError as e:
            raise ValueError(
                f"event {event['event_no']}: sensor key {e} is not in "
                "the geometry -- data and geometry asset disagree"
            ) from e
        res = sample_event(
            p[:, lay.x],
            p[:, lay.y],
            p[:, lay.z],
            p[:, lay.t],
            p[:, lay.charge],
            self.geo,
            rng,
            sensor,
            holdout_mode=self.holdout_mode,
            random_vis_frac=self.random_vis_frac,
            q_lo=self.q_lo,
            q_hi=self.q_hi,
            pos_k=self.pos_k,
            neg_anchor=self.neg_anchor,
            rand_neg_frac=self.rand_neg_frac,
            min_visible=self.min_visible,
            min_future=self.min_future,
            resample_tries=self.resample_tries,
        )
        if res is None:
            # the sampler's deterministic fallback guarantees a split for any
            # event with enough hit sensors, so None means under-filtered input
            raise ValueError(
                f"event {event['event_no']} has too few hit sensors for a "
                f"{self.holdout_mode!r} split (needs >= min_visible + "
                f"min_future = {self.min_visible + self.min_future}); "
                "pre-filter the selection to usable events"
            )
        vis = p[res["vis_pulse_mask"]]
        if len(vis) < 2:  # unreachable for min_visible >= 2
            raise ValueError(
                f"event {event['event_no']}: split left {len(vis)} visible "
                "pulses; set min_visible >= 2"
            )
        if self.center_time:
            vis = vis.copy()
            vis[:, lay.t] -= res["t_cwm"]
        if len(vis) > self.max_pulses:
            vis = vis[rng.choice(len(vis), self.max_pulses, replace=False)]
        return dict(
            vis=vis.astype(np.float32),
            qpos=res["query_pos"].astype(np.float32),
            label=res["query_label"].astype(np.float32),
            dt=(res["query_dt"] / self.dt_scale).astype(np.float32),
            # pos/nearest-dark queries vs random dark DOMs; consumed by the
            # easy-vs-hard split of the validation AUCs
            hard=(res["query_tag"] != "rand").astype(np.float32),
        )

    def collate(self, samples: list[Sample]) -> dict:
        """Standardize per event and pack jagged NJTs (pulses + queries).

        Args:
            samples: The batch's samples as make_sample built them.

        Returns:
            Batch dict of jagged nested tensors: pulses, qpos, label, dt,
            hard.
        """

        def jag(tensors):
            return torch.nested.nested_tensor(tensors, layout=torch.jagged)

        # standardize per event, then pack ragged -- no padding decision here
        pulses = jag(
            [self.scaler.scale_pulses(torch.from_numpy(s["vis"])) for s in samples]
        )
        qpos = jag(
            [self.scaler.scale_positions(torch.from_numpy(s["qpos"])) for s in samples]
        )
        label = jag([torch.from_numpy(s["label"]) for s in samples])
        dt = jag([torch.from_numpy(s["dt"]) for s in samples])
        hard = jag([torch.from_numpy(s["hard"]) for s in samples])
        return dict(pulses=pulses, qpos=qpos, label=label, dt=dt, hard=hard)

    # ---- model side (GPU) -----------------------------------------------
    def build_head(self, dim: int) -> nn.Module:
        """Build the shared query encoder with one head per objective.

        Args:
            dim: The backbone's embedding width.

        Returns:
            The MultiObjectiveHead consuming query positions + encoded events.
        """
        return MultiObjectiveHead(dim, self.objectives)

    def loss(self, preds: list[Tensor], batch: dict) -> tuple[Tensor, dict[str, float]]:
        """Sum the objectives' weighted losses over the real queries.

        Args:
            preds: One [B, Qmax, C_i] prediction tensor per objective.
            batch: The collated batch holding the targets.

        Returns:
            Total loss and per-objective {loss_<name>: value} metrics.
        """
        # `preds` is one [B, Qmax, C_i] per objective (same order). Real queries
        # pack at the front of each row (to_padded), so a mask from the query
        # NJT's per-event lengths selects them; each objective scores its own flat
        # [sum_Q, C_i] and owns its target lookup + masking.
        qmask = real_query_mask(preds[0], batch)
        total = preds[0].new_zeros(())
        metrics: dict[str, float] = {}
        for obj, pred in zip(self.objectives, preds, strict=True):
            term = obj.loss(pred[qmask], batch)
            total = total + obj.weight * term
            metrics[f"loss_{obj.name}"] = float(term.detach())
        return total, metrics
