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
from spine.pretext.curtain.head import MultiObjectiveHead
from spine.pretext.curtain.sampler import SamplerConfig, sample_event
from spine.pretext.registry import TASKS


@TASKS.register("curtain")
class CurtainTask(PretextTask):
    def __init__(self, geo: dict, objectives: List[Objective],
                 scaler: FeatureScaler,
                 sampler_cfg: Optional[SamplerConfig] = None,
                 max_pulses: int = 768, center_time: bool = True,
                 dt_scale: float = 500.0):
        self.geo = geo
        self.objectives = objectives
        self.scaler = scaler
        self.cfg = sampler_cfg or SamplerConfig()
        self.max_pulses = max_pulses
        self.center_time = center_time
        self.dt_scale = dt_scale

    # ---- data side (CPU, per event / per batch) -------------------------
    def make_sample(self, event: Dict[str, np.ndarray],
                    rng: np.random.Generator) -> Optional[Sample]:
        p = event["pulses"]  # [P, n] raw; columns per self.scaler.layout
        lay = self.scaler.layout
        res = sample_event(p[:, lay.x], p[:, lay.y], p[:, lay.z],
                           p[:, lay.t], p[:, lay.charge],
                           self.geo, self.cfg, rng)
        if res is None:
            return None
        vis = p[res["vis_pulse_mask"]]
        if len(vis) < 2:
            return None
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
        )

    def collate(self, samples: List[Optional[Sample]]):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None

        def jag(tensors):
            return torch.nested.nested_tensor(tensors, layout=torch.jagged)

        # standardize per event, then pack ragged -- no padding decision here
        pulses = jag([self.scaler.scale_pulses(torch.from_numpy(s["vis"]))
                      for s in samples])
        qpos = jag([self.scaler.scale_positions(torch.from_numpy(s["qpos"]))
                    for s in samples])
        label = jag([torch.from_numpy(s["label"]) for s in samples])
        dt = jag([torch.from_numpy(s["dt"]) for s in samples])
        return dict(pulses=pulses, qpos=qpos, label=label, dt=dt)

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
