"""Pretext-task interface -- the extension point.

A pretext task defines a self-supervised objective over raw events. It owns:

  * make_sample(event, rng) -> Sample | None
        CPU, per-event: build the encoder input + targets. This is where
        masking / view generation / target construction lives (for CURTAIN: the
        random time-cut split + query sampling). Returns None for an unusable
        event; the datamodule substitutes another so a batch is never empty
        (an all-empty batch makes a DDP rank skip its step and deadlock).
  * collate(samples) -> Batch
        variable-length padding into a batch container.
  * build_head(dim) -> nn.Module
        the prediction head(s) on top of the backbone's token embeddings.
  * loss(head_out, batch) -> (scalar loss, metrics)
        score the head output against the sample's targets.

A task may carry several weighted `Objective`s over one shared sample -- which
is exactly the CURTAIN v1 -> v2 relationship (add the Delta-t objective). So
"v2" is an added objective in config, not a forked task/model/dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from torch import Tensor, nn

Sample = Dict[str, Any]   # per-event: raw encoder input + targets
Batch = Any               # collated batch (task-defined container)


@dataclass
class Objective:
    """One weighted prediction target over a task's sample.

    `channels` is how many head-output channels this objective consumes; a task
    with objectives [occupancy(1), dt(1)] builds a 2-channel head. `loss_fn`
    maps (pred_channels, target) -> scalar.
    """

    name: str
    channels: int
    target_key: str
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    weight: float = 1.0
    metric_fn: Optional[Callable] = None


class PretextTask(ABC):
    """Factory + transform + loss for one self-supervised objective."""

    #: objectives this task scores (defines head width and the loss terms)
    objectives: List[Objective]

    @abstractmethod
    def make_sample(self, event: Dict[str, np.ndarray],
                    rng: np.random.Generator) -> Optional[Sample]:
        raise NotImplementedError

    @abstractmethod
    def collate(self, samples: List[Sample]) -> Batch:
        raise NotImplementedError

    @abstractmethod
    def build_head(self, dim: int) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def loss(self, head_out: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, float]]:
        raise NotImplementedError
