"""DataModule: raw event -> pretext sample, with per-epoch re-randomization.

Each __getitem__ reads a raw event and runs `task.make_sample` with a FRESH RNG
(so the pretext augmentation re-randomizes every epoch; val uses a fixed seed).
It never yields None -- it substitutes the next usable event, because an
all-unusable batch makes a DDP rank skip its step and deadlock the collectives.
Batching is `task.collate`.
"""

from __future__ import annotations

from typing import List, Optional

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from spine.data.sources import EventSource
from spine.pretext.base import PretextTask


class PretextDataset(Dataset):
    def __init__(self, source: EventSource, event_nos: List[int],
                 task: PretextTask, augment: bool = True):
        self.source = source
        self.ev = list(event_nos)
        self.task = task
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ev)

    def __getitem__(self, idx: int):
        for off in range(len(self.ev)):
            j = (idx + off) % len(self.ev)
            event = self.source.read(self.ev[j])
            if event["pulses"].shape[0] < 4:
                continue
            rng = np.random.default_rng(None if self.augment else j)
            sample = self.task.make_sample(event, rng)
            if sample is not None:
                return sample
        raise RuntimeError("no usable event in the entire selection")


class SpineDataModule(pl.LightningDataModule):
    def __init__(self, source: EventSource, task: PretextTask,
                 train_events, val_events, batch_size: int = 64,
                 num_workers: int = 16):
        super().__init__()
        self.source = source
        self.task = task
        self.train_events = train_events
        self.val_events = val_events
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _loader(self, events, augment: bool, shuffle: bool) -> DataLoader:
        return DataLoader(
            PretextDataset(self.source, events, self.task, augment=augment),
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, collate_fn=self.task.collate,
            persistent_workers=self.num_workers > 0, drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_events, augment=True, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_events, augment=False, shuffle=False)
