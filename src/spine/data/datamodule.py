"""DataModule: raw event -> pretext sample.

`__getitem__` is a pure index -> sample map. It fails loud if an event has too
few pulses -- the caller is responsible for supplying a pre-filtered, usable
selection (see spine.data.selection.filter_by_min_count). It never returns None
and never substitutes: `task.make_sample` is guaranteed to produce a split for
any event that passed the filter (the sampler's deterministic fallback), so a
batch is never empty (an empty batch makes a DDP rank skip its step and
deadlock). Batching is `task.collate`.

Staging the DB to node-local scratch and building the frozen split belong in
prepare_data()/setup() (TODO -- pulls that logic out of the sbatch).
"""

from __future__ import annotations

from typing import List

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from spine.data.sources import EventSource
from spine.pretext.base import PretextTask


class PretextDataset(Dataset):
    def __init__(self, source: EventSource, event_nos: List[int],
                 task: PretextTask, augment: bool = True, min_pulses: int = 4):
        self.source = source
        self.ev = list(event_nos)
        self.task = task
        self.augment = augment
        self.min_pulses = min_pulses

    def __len__(self) -> int:
        return len(self.ev)

    def __getitem__(self, idx: int):
        ev = self.ev[idx]
        event = self.source.read(ev)
        n = int(event["pulses"].shape[0])
        if n < self.min_pulses:
            raise ValueError(
                f"event {ev} has {n} pulses (< {self.min_pulses}); pre-filter "
                "the selection so __getitem__ stays a pure index -> sample map"
            )
        rng = np.random.default_rng(None if self.augment else idx)
        sample = self.task.make_sample(event, rng)
        if sample is None:
            raise ValueError(
                f"event {ev} passed the pulse check but make_sample could not "
                "form a split (too few hit sensors); tighten the selection "
                "filter (min hit sensors >= min_visible + min_future)"
            )
        return sample


class SpineDataModule(pl.LightningDataModule):
    def __init__(self, source: EventSource, task: PretextTask,
                 train_events, val_events, batch_size: int = 64,
                 num_workers: int = 16, min_pulses: int = 4):
        super().__init__()
        self.source = source
        self.task = task
        self.train_events = train_events
        self.val_events = val_events
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_pulses = min_pulses

    def _loader(self, events, augment: bool, shuffle: bool) -> DataLoader:
        return DataLoader(
            PretextDataset(self.source, events, self.task, augment=augment,
                           min_pulses=self.min_pulses),
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, collate_fn=self.task.collate,
            persistent_workers=self.num_workers > 0, drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_events, augment=True, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_events, augment=False, shuffle=False)
