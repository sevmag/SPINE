"""CurtainTask -- assembles sampler + head + objectives into a PretextTask.

make_sample runs the CURTAIN sampler on RAW pulses (fresh RNG per call = a new
split each epoch); collate pads to a batch and standardizes at the model
boundary; build_head sizes the head to the objectives' total channels; loss
scores each objective with the right mask (occupancy over all valid queries,
dt over hit queries only) and weight.
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
        p = event["pulses"]  # [P,5] raw (x,y,z,t,charge)
        res = sample_event(p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4],
                           self.geo, self.cfg, rng)
        if res is None:
            return None
        vis = p[res["vis_pulse_mask"]]
        if len(vis) < 2:
            return None
        if self.center_time:
            vis = vis.copy()
            vis[:, 3] -= res["t_cwm"]
        if len(vis) > self.max_pulses:
            vis = vis[rng.choice(len(vis), self.max_pulses, replace=False)]
        return dict(
            vis=vis.astype(np.float32),
            qpos=res["query_pos"].astype(np.float32),
            label=res["query_label"].astype(np.float32),
            dt=(res["query_dt"] / self.dt_scale).astype(np.float32),
            hard=(res["query_tag"] != "rand"),
        )

    def collate(self, samples: List[Optional[Sample]]):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        n = len(samples)
        lmax = max(len(s["vis"]) for s in samples)
        qmax = max(len(s["label"]) for s in samples)
        x = torch.zeros(n, lmax, 5)
        tok_mask = torch.zeros(n, lmax, dtype=torch.bool)
        qpos = torch.zeros(n, qmax, 3)
        label = torch.zeros(n, qmax)
        dt = torch.zeros(n, qmax)
        valid = torch.zeros(n, qmax, dtype=torch.bool)
        hard = torch.zeros(n, qmax, dtype=torch.bool)
        for i, s in enumerate(samples):
            lp, lq = len(s["vis"]), len(s["label"])
            x[i, :lp] = self.scaler.scale_pulses(torch.from_numpy(s["vis"]))
            tok_mask[i, :lp] = True
            qpos[i, :lq] = self.scaler.scale_positions(torch.from_numpy(s["qpos"]))
            label[i, :lq] = torch.from_numpy(s["label"])
            dt[i, :lq] = torch.from_numpy(s["dt"])
            valid[i, :lq] = True
            hard[i, :lq] = torch.from_numpy(s["hard"])
        return dict(x=x, token_mask=tok_mask, qpos=qpos, label=label, dt=dt,
                    valid=valid, hard=hard)

    # ---- model side (GPU) -----------------------------------------------
    def build_head(self, dim: int) -> nn.Module:
        channels = sum(o.channels for o in self.objectives)
        return QueryCrossAttnHead(dim, channels)

    def loss(self, head_out: Tensor, batch: dict) -> Tuple[Tensor, Dict[str, float]]:
        valid = batch["valid"]
        total = head_out.new_zeros(())
        metrics: Dict[str, float] = {}
        ch = 0
        for obj in self.objectives:
            pred = head_out[..., ch:ch + obj.channels]
            ch += obj.channels
            if obj.name == "dt":
                # regress Delta-t on hit queries only (label == 1)
                m = valid & (batch["label"] > 0.5)
                if m.any():
                    term = obj.loss_fn(pred[m], batch["dt"][m])
                else:
                    term = head_out.new_zeros(())
            else:
                term = obj.loss_fn(pred[valid], batch[obj.target_key][valid])
            total = total + obj.weight * term
            metrics[f"loss_{obj.name}"] = float(term.detach())
        return total, metrics
