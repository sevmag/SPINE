"""DataModule: a raw-pulse read Dataset -> pretext samples.

SPINE is reader-agnostic and ships no reader. A read Dataset satisfies the
RawPulseDataset contract:

    raw[i] -> {"event_no": int, "pulses": np.ndarray[P, F]}   # F RAW feature cols

Build one with graphnet's LMDBDataset / SQLiteDataset (recommended -- see
examples/readers.py for the adapter) or bring your own. Pulses must be RAW
(SPINE standardizes AFTER the pretext split); the number and order of the F
feature columns are not fixed here -- they are defined by the task's
`FeatureScaler` layout (`spine.data.scaling.FeatureLayout`), and the reader must
emit columns in that order.

PretextDataset composes on top of any such Dataset: read by index, fail loud on
a wrong shape or too few pulses (pre-filter the selection --
selection.filter_by_min_count), then run task.make_sample with a fresh RNG when
`resample` (train) / a fixed per-item seed otherwise (val). It never returns
None and never substitutes -- make_sample is guaranteed to split any filtered
event -- so a batch is never empty (an empty batch deadlocks DDP). Batching is
task.collate.

Staging the read store and building the frozen split belong in
prepare_data()/setup() (TODO -- pulls that out of the sbatch).
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from spine.pretext.base import PretextTask


class RawEvent(TypedDict):
    event_no: int
    pulses: np.ndarray  # [P, F] raw; column order per the task's FeatureLayout


@runtime_checkable
class RawPulseDataset(Protocol):
    """The read-layer contract SPINE consumes (any Dataset that satisfies it)."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> RawEvent: ...


class PretextDataset(Dataset):
    """Transform on top of a read Dataset: index -> pretext sample."""

    def __init__(self, raw: RawPulseDataset, task: PretextTask,
                 resample: bool = True, min_pulses: int = 4):
        self.raw = raw
        self.task = task
        self.resample = resample
        self.min_pulses = min_pulses

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        event = self.raw[idx]  # plain index lookup on the read Dataset
        pulses = np.asarray(event["pulses"])
        n = int(pulses.shape[0])
        if n < self.min_pulses:
            raise ValueError(
                f"event {event['event_no']} has {n} pulses (< {self.min_pulses}); "
                "pre-filter the selection so __getitem__ stays index -> sample"
            )
        if pulses.ndim != 2:
            raise ValueError(
                f"read Dataset broke the contract for event {event['event_no']}: "
                f"pulses must be a 2-D [P, F] raw array (feature columns in the "
                f"task's FeatureLayout order), got shape {tuple(pulses.shape)}"
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
    def __init__(self, train_raw: RawPulseDataset, val_raw: RawPulseDataset,
                 task: PretextTask, batch_size: int = 64, num_workers: int = 16,
                 min_pulses: int = 4):
        super().__init__()
        self.train_raw = train_raw
        self.val_raw = val_raw
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_pulses = min_pulses

    def _loader(self, raw: RawPulseDataset, resample: bool, shuffle: bool) -> DataLoader:
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
