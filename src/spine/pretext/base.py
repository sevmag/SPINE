"""Pretext-task interface -- the extension point.

A pretext task defines a self-supervised objective over raw events. It owns:

  * make_sample(event, rng) -> Sample
        CPU, per-event: build the encoder input + targets. This is where
        masking / view generation / target construction lives (for CURTAIN: the
        random time-cut split + query sampling). MUST return a Sample or raise:
        an event the task cannot handle is caller error (pre-filter the
        selection), never a silent skip -- skipping would allow an empty batch,
        and an all-empty batch makes a DDP rank skip its step and deadlock.
  * collate(samples) -> Batch
        pack variable-length samples into a batch container (task-defined; the
        CURTAIN task uses jagged nested tensors, projected by backbone/head).
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
from typing import Any, Dict, List, Tuple

import numpy as np
from torch import Tensor, nn

Sample = Dict[str, Any]   # per-event: raw encoder input + targets
Batch = Any               # collated batch (task-defined container)


class Objective(ABC):
    """One weighted prediction target: its own head + loss.

    Each objective owns everything specific to it -- the head that maps the
    shared per-query embedding to predictions, and the loss (target lookup +
    masking + reduction) over the real queries. Adding an objective is a
    subclass; the task just sums `weight * objective.loss(...)`, with no
    per-objective branching.
    """

    name: str

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def build_head(self, dim: int) -> nn.Module:
        """Shared per-query embedding [.., dim] -> this objective's prediction."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred: Tensor, batch: dict) -> Tensor:
        """Scalar loss; `pred` is [sum_Q, channels] over the real queries."""
        raise NotImplementedError


class PretextTask(ABC):
    """Factory + transform + loss for one self-supervised objective."""

    #: objectives this task scores (defines head width and the loss terms)
    objectives: List[Objective]

    @abstractmethod
    def make_sample(self, event: Dict[str, np.ndarray],
                    rng: np.random.Generator) -> Sample:
        raise NotImplementedError

    @abstractmethod
    def collate(self, samples: List[Sample]) -> Batch:
        raise NotImplementedError

    @abstractmethod
    def build_head(self, dim: int) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def loss(self, output, batch: Batch) -> Tuple[Tensor, Dict[str, float]]:
        raise NotImplementedError

    # ---- optional epoch-level validation metrics ------------------------
    # A task may accumulate CPU-side state per val step and reduce it to
    # metrics once per epoch (e.g. CURTAIN's occupancy AUCs, which need the
    # full epoch's scores, not a per-batch mean). Default: no metrics.
    def val_step_cache(self, output, batch: Batch):
        """Small CPU payload to keep from one val step (None = keep nothing)."""
        return None

    def val_epoch_metrics(self, caches: List[Any]) -> Dict[str, float]:
        """Reduce the epoch's cached payloads to {metric_name: value}."""
        return {}
