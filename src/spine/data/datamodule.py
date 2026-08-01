"""Raw-pulse read Datasets -> pretext samples.

THE read contract: raw[i] -> {"event_no": int, "pulses": [P, F] raw,
"sensor_key": [P] int} -- feature columns per the task's FeatureLayout, raw
values (standardization happens after the pretext split), sensor keys matching
the geometry asset's key array (multi-level IDs composed by the reader;
single-PMT detectors use 1 for the missing level). SPINE ships no reader;
examples/readers.py shows two. PretextDataset is a pure index -> sample map --
make_sample raises on events it cannot use, so batches are never silently
short (an empty batch deadlocks DDP).
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from spine.pretext.base import PretextTask


class RawEvent(TypedDict):
    """One raw event exactly as the read layer returns it."""

    event_no: int
    pulses: np.ndarray  # [P, F] raw; column order per the task's FeatureLayout
    sensor_key: np.ndarray  # [P] int sensor identity per pulse (see module doc)


class RawPulseDataset(Protocol):
    """The read-layer contract SPINE consumes (any Dataset that satisfies it)."""

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> RawEvent: ...


class PretextDataset(Dataset):
    """Transform on top of a read Dataset: index -> pretext sample."""

    def __init__(
        self,
        raw: RawPulseDataset,
        task: PretextTask,
        resample: bool = True,
    ):
        """Compose the pretext transform over a read Dataset.

        Args:
            raw: Read-layer Dataset satisfying the RawPulseDataset contract.
            task: Pretext task whose make_sample transforms each event.
            resample: Fresh RNG per call (training) instead of a fixed
                per-index seed (validation).
        """
        self.raw = raw
        self.task = task
        self.resample = resample

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        event = self.raw[idx]  # plain index lookup on the read Dataset
        pulses = np.asarray(event["pulses"])
        if pulses.ndim != 2:
            raise ValueError(
                f"read Dataset broke the contract for event {event['event_no']}: "
                f"pulses must be a 2-D [P, F] raw array (feature columns in the "
                f"task's FeatureLayout order), got shape {tuple(pulses.shape)}"
            )
        rng = np.random.default_rng(None if self.resample else idx)
        return self.task.make_sample(event, rng)  # raises on unsplittable events


class SpineDataModule(pl.LightningDataModule):
    """Train/val DataLoaders over PretextDataset with the task's collate."""

    def __init__(
        self,
        train_raw: RawPulseDataset,
        val_raw: RawPulseDataset,
        task: PretextTask,
        batch_size: int = 64,
        num_workers: int = 16,
        val_num_workers: int | None = None,
    ):
        """Hold the two read Datasets and the loader settings.

        Args:
            train_raw: Read Dataset for the training events.
            val_raw: Read Dataset for the validation events.
            task: Pretext task providing make_sample and collate.
            batch_size: Events per batch for both loaders.
            num_workers: Worker processes for the training loader.
            val_num_workers: Worker processes for the validation loader;
                None uses num_workers.
        """
        super().__init__()
        self.train_raw = train_raw
        self.val_raw = val_raw
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = (
            num_workers if val_num_workers is None else val_num_workers
        )

    def _loader(
        self, raw: RawPulseDataset, resample: bool, shuffle: bool, workers: int
    ) -> DataLoader:
        # spawn, not fork: forking loader workers from a process that already
        # runs NCCL/CUDA threads can deadlock a DDP rank; spawn children start
        # clean, and persistent workers pay the startup cost once.
        return DataLoader(
            PretextDataset(raw, self.task, resample=resample),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=self.task.collate,
            persistent_workers=workers > 0,
            multiprocessing_context="spawn" if workers > 0 else None,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled drop-last loader; a fresh pretext split every epoch.

        Returns:
            The training DataLoader.
        """
        return self._loader(
            self.train_raw, resample=True, shuffle=True, workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Deterministic loader; per-event fixed RNG seeds.

        Returns:
            The validation DataLoader.
        """
        return self._loader(
            self.val_raw, resample=False, shuffle=False, workers=self.val_num_workers
        )
