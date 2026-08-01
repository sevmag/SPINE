"""Pretext-task interface: make_sample -> collate -> build_head -> loss.

A task owns its per-event sampling, batching, head construction and loss;
`Objective`s are its weighted sub-targets, each bringing its own head and
masking. Adding an SSL method is one new task plugin -- data, backbone and
engine stay unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from torch import Tensor, nn

Sample = dict[str, Any]  # per-event: raw encoder input + targets
Batch = Any  # collated batch (task-defined container)


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
        """Set this objective's loss weight.

        Args:
            weight: Multiplier on this objective's loss in the task total.
        """
        self.weight = weight

    @abstractmethod
    def build_head(self, dim: int) -> nn.Module:
        """Build this objective's prediction head.

        Args:
            dim: Width of the shared per-query embedding the head consumes.

        Returns:
            Module mapping [..., dim] embeddings to this objective's outputs.
        """
        ...

    @abstractmethod
    def loss(self, pred: Tensor, batch: dict) -> Tensor:
        """Score this objective's predictions.

        Args:
            pred: [sum_Q, channels] predictions over the real queries.
            batch: The collated batch holding this objective's targets.

        Returns:
            Scalar loss.
        """
        ...


class PretextTask(ABC):
    """Factory + transform + loss for one self-supervised objective."""

    #: objectives this task scores (defines head width and the loss terms)
    objectives: list[Objective]

    @abstractmethod
    def make_sample(
        self, event: dict[str, np.ndarray], rng: np.random.Generator
    ) -> Sample:
        """Build one event's Sample.

        Args:
            event: One raw event from the read layer.
            rng: Per-call generator; fresh entropy resamples the pretext,
                a fixed seed reproduces it.

        Returns:
            The per-event sample (encoder input + targets). An unusable event
            must raise, never be silently skipped.
        """
        ...

    @abstractmethod
    def collate(self, samples: list[Sample]) -> Batch:
        """Pack per-event Samples into the batch container the model sees.

        Args:
            samples: The events of one batch, as make_sample built them.

        Returns:
            The task-defined batch container.
        """
        ...

    @abstractmethod
    def build_head(self, dim: int) -> nn.Module:
        """Construct the prediction head for a `dim`-wide backbone.

        Args:
            dim: The backbone's embedding width.

        Returns:
            The head module the engine calls on encoded batches.
        """
        ...

    @abstractmethod
    def loss(self, output: Any, batch: Batch) -> tuple[Tensor, dict[str, float]]:
        """Score the head output against the batch's targets.

        Args:
            output: What the head produced for this batch.
            batch: The collated batch holding the targets.

        Returns:
            Total loss and a {name: value} metrics dict.
        """
        ...
