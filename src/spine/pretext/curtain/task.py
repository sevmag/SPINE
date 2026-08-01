"""CurtainTask -- assembles sampler + head + objectives into a PretextTask.

make_sample runs the CURTAIN sampler on RAW pulses (fresh RNG per call = a new
split each epoch); collate standardizes each event and packs the batch as jagged
nested tensors (pulses + query fields) with no padding baked in -- the backbone
and head project via to_padded_tensor; build_head sizes the head to the
build_head assembles the shared query encoder with one head per objective; loss
sums each objective's own loss (which owns its target lookup + masking) over the
real queries, weighted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from spine.data.scaling import FeatureScaler
from spine.pretext.base import Objective, PretextTask, Sample
from spine.utils import auc
from spine.pretext.curtain.head import MultiObjectiveHead
from spine.pretext.curtain.sampler import SamplerConfig, sample_event


class CurtainTask(PretextTask):
    def __init__(self, geo: dict, objectives: List[Objective],
                 scaler: FeatureScaler,
                 sampler: Optional[SamplerConfig] = None,
                 max_pulses: int = 768, center_time: bool = True,
                 dt_scale: float = 500.0):
        self.geo = geo
        self.objectives = objectives
        self.scaler = scaler
        self.sampler = sampler or SamplerConfig()
        self.max_pulses = max_pulses
        self.center_time = center_time
        self.dt_scale = dt_scale

    # ---- data side (CPU, per event / per batch) -------------------------
    def make_sample(self, event: Dict[str, np.ndarray],
                    rng: np.random.Generator) -> Sample:
        p = event["pulses"]  # [P, n] raw; columns per self.scaler.layout
        lay = self.scaler.layout
        res = sample_event(p[:, lay.x], p[:, lay.y], p[:, lay.z],
                           p[:, lay.t], p[:, lay.charge],
                           self.geo, self.sampler, rng)
        if res is None:
            # the sampler's deterministic fallback guarantees a split for any
            # event with enough hit sensors, so None means under-filtered input
            raise ValueError(
                f"event {event['event_no']} is not splittable in "
                f"{self.sampler.holdout_mode!r} mode: fewer than min_visible + "
                f"min_future = "
                f"{self.sampler.min_visible + self.sampler.min_future} hit "
                "sensors, or a degenerate first-hit-time spread; pre-filter "
                "the selection with sampler.can_always_split"
            )
        vis = p[res["vis_pulse_mask"]]
        if len(vis) < 2:  # unreachable for min_visible >= 2
            raise ValueError(
                f"event {event['event_no']}: split left {len(vis)} visible "
                "pulses; set sampler.min_visible >= 2"
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
            # pos/hard-negative queries vs random dark DOMs -- only consumed by
            # the val AUC split (easy vs hard negatives)
            hard=(res["query_tag"] != "rand").astype(np.float32),
        )

    def collate(self, samples: List[Sample]):
        def jag(tensors):
            return torch.nested.nested_tensor(tensors, layout=torch.jagged)

        # standardize per event, then pack ragged -- no padding decision here
        pulses = jag([self.scaler.scale_pulses(torch.from_numpy(s["vis"]))
                      for s in samples])
        qpos = jag([self.scaler.scale_positions(torch.from_numpy(s["qpos"]))
                    for s in samples])
        label = jag([torch.from_numpy(s["label"]) for s in samples])
        dt = jag([torch.from_numpy(s["dt"]) for s in samples])
        hard = jag([torch.from_numpy(s["hard"]) for s in samples])
        return dict(pulses=pulses, qpos=qpos, label=label, dt=dt, hard=hard)

    # ---- model side (GPU) -----------------------------------------------
    def build_head(self, dim: int) -> nn.Module:
        return MultiObjectiveHead(dim, self.objectives)

    def loss(self, preds, batch: dict) -> Tuple[Tensor, Dict[str, float]]:
        # `preds` is one [B, Qmax, C_i] per objective (same order). Real queries
        # pack at the front of each row (to_padded), so a mask from the query
        # NJT's per-event lengths selects them; each objective scores its own flat
        # [sum_Q, C_i] and owns its target lookup + masking.
        qlen = batch["qpos"].offsets().diff()
        qmask = (torch.arange(preds[0].shape[1], device=preds[0].device)[None]
                 < qlen[:, None])
        total = preds[0].new_zeros(())
        metrics: Dict[str, float] = {}
        for obj, pred in zip(self.objectives, preds):
            term = obj.loss(pred[qmask], batch)
            total = total + obj.weight * term
            metrics[f"loss_{obj.name}"] = float(term.detach())
        return total, metrics

    # ---- epoch-level val metrics: occupancy AUC over the full val set ----
    def val_step_cache(self, preds, batch: dict):
        occ = next((i for i, o in enumerate(self.objectives)
                    if o.name == "occupancy"), None)
        if occ is None:
            return None
        qlen = batch["qpos"].offsets().diff()
        qmask = (torch.arange(preds[occ].shape[1],
                              device=preds[occ].device)[None] < qlen[:, None])
        return (preds[occ][qmask].squeeze(-1).detach().float().cpu().numpy(),
                batch["label"].values().cpu().numpy(),
                batch["hard"].values().cpu().numpy() > 0.5)

    def val_epoch_metrics(self, caches):
        lg = np.concatenate([c[0] for c in caches])
        y = np.concatenate([c[1] for c in caches])
        hd = np.concatenate([c[2] for c in caches])
        easy = (y == 1) | (~hd)  # positives + random(easy) negatives
        return {"val_auc_all": auc(lg, y),
                "val_auc_hard": auc(lg[hd], y[hd]),
                "val_auc_easy": auc(lg[easy], y[easy])}
