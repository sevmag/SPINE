"""CurtainTask -- assembles sampler + head + objectives into a PretextTask.

make_sample runs the CURTAIN sampler on RAW pulses (fresh RNG per call = a new
split each epoch); collate standardizes each event and packs the batch as jagged
nested tensors (pulses + query fields) with no padding baked in -- the backbone
and head project via to_padded_tensor; build_head sizes the head to the
objectives' total channels; loss scores each objective over the real queries
(the query NJT's values) with the right mask (occupancy over all queries, dt over
hit queries only) and weight.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from spine.data.scaling import FeatureScaler
from spine.pretext.base import Objective, PretextTask, Sample
from spine.pretext.curtain.head import QueryCrossAttnHead
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
        channels = sum(o.channels for o in self.objectives)
        return QueryCrossAttnHead(dim, channels)

    def loss(self, head_out: Tensor, batch: dict) -> Tuple[Tensor, Dict[str, float]]:
        # head_out is [B, Qmax, C] over PADDED queries; the real queries pack at
        # the front of each row, so a mask from the query NJT's per-event lengths
        # selects them, and their targets are the NJT's flat values (same order).
        qlen = batch["qpos"].offsets().diff()
        qmask = (torch.arange(head_out.shape[1], device=head_out.device)[None]
                 < qlen[:, None])
        pred = head_out[qmask]                 # [sum_Q, C]
        label = batch["label"].values()         # [sum_Q]
        total = head_out.new_zeros(())
        metrics: Dict[str, float] = {}
        ch = 0
        for obj in self.objectives:
            p = pred[:, ch:ch + obj.channels]
            ch += obj.channels
            target = batch[obj.target_key].values()
            if obj.name == "dt":
                m = label > 0.5   # regress Delta-t on hit queries only
                term = (obj.loss_fn(p[m], target[m]) if m.any()
                        else head_out.new_zeros(()))
            else:
                term = obj.loss_fn(p, target)
            total = total + obj.weight * term
            metrics[f"loss_{obj.name}"] = float(term.detach())
        return total, metrics
