"""DataModule: a raw-pulse Dataset -> pretext samples.

PretextDataset composes on top of any read Dataset (`raw[idx] -> {event_no,
pulses}`): it reads by index, fails loud if an event has too few pulses (the
caller supplies a pre-filtered selection -- see selection.filter_by_min_count),
and runs task.make_sample with a fresh RNG (resampled per epoch on train; a fixed per-item seed on val).
It never returns None and never substitutes -- make_sample is guaranteed to
split any filtered event -- so a batch is never empty (an empty batch deadlocks
DDP). Batching is task.collate.

Staging the DB to node-local scratch and building the frozen split belong in
prepare_data()/setup() (TODO -- pulls that out of the sbatch).
"""

from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from spine.pretext.base import PretextTask


class PretextDataset(Dataset):
    """Transform on top of a raw read Dataset: index -> pretext sample."""

    def __init__(self, raw: Dataset, task: PretextTask, resample: bool = True,
                 min_pulses: int = 4):
        self.raw = raw
        self.task = task
        self.resample = resample
        self.min_pulses = min_pulses

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        event = self.raw[idx]  # plain index lookup on the read Dataset
        n = int(event["pulses"].shape[0])
        if n < self.min_pulses:
            raise ValueError(
                f"event {event['event_no']} has {n} pulses (< {self.min_pulses}); "
                "pre-filter the selection so __getitem__ stays index -> sample"
            )
        rng = np.random.default_rng(None if self.resample else idx)
        sample = self.task.make_sample(event, rng)
        if sample is None:
            raise ValueError(
                f"event {event['event_no']} passed the pulse check but "
                "make_sample could not split it (too few hit sensors); tighten "
                "the selection filter (min hit sensors >= min_visible + min_future)"
            )
        return sample


class SpineDataModule(pl.LightningDataModule):
    def __init__(self, train_raw: Dataset, val_raw: Dataset, task: PretextTask,
                 batch_size: int = 64, num_workers: int = 16, min_pulses: int = 4):
        super().__init__()
        self.train_raw = train_raw
        self.val_raw = val_raw
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_pulses = min_pulses

    def _loader(self, raw: Dataset, resample: bool, shuffle: bool) -> DataLoader:
        return DataLoader(
            PretextDataset(raw, self.task, resample=resample,
                           min_pulses=self.min_pulses),
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, collate_fn=self.task.collate,
            persistent_workers=self.num_workers > 0, drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_raw, resample=True, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_raw, resample=False, shuffle=False)
