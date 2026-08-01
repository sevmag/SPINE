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

Readers must also emit `sensor_key`: one integer per pulse, unique per
physical sensor, matching a per-row key array of the geometry asset
(load_geometry(sensor_key=...)). Sensor identity always comes from the data
-- reconstructing it from coordinates is not supported, since float matching
collapses near-duplicate positions at the margins. Multi-level IDs
(string / module / PMT) are composed by the reader into one integer;
single-PMT detectors use the constant 1 for the missing PMT level.

PretextDataset composes on top of any such Dataset: read by index, fail loud on
a wrong shape or too few pulses (pre-filter your selection to usable events),
then run task.make_sample with a fresh RNG when
`resample` (train) / a fixed per-item seed otherwise (val). make_sample raises
on any event it cannot split (never a silent skip or substitution), so a batch
is never empty (an empty batch deadlocks DDP). Batching is task.collate.

Staging the read store and building the frozen split belong in
prepare_data()/setup() (TODO -- pulls that out of the sbatch).
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from spine.pretext.base import PretextTask


class RawEvent(TypedDict):
    """One raw event exactly as the read layer returns it."""

    event_no: int
    pulses: np.ndarray  # [P, F] raw; column order per the task's FeatureLayout
    sensor_key: np.ndarray  # [P] int sensor identity per pulse (see module doc)


@runtime_checkable
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
        min_pulses: int = 4,
    ):
        """Compose the pretext transform over a read Dataset.

        Args:
            raw: Read-layer Dataset satisfying the RawPulseDataset contract.
            task: Pretext task whose make_sample transforms each event.
            resample: Fresh RNG per call (training) instead of a fixed
                per-index seed (validation).
            min_pulses: Fail-loud floor on the raw pulse count per event.
        """
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
        min_pulses: int = 4,
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
            min_pulses: Fail-loud floor on the raw pulse count per event.
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
        self.min_pulses = min_pulses

    def _loader(
        self, raw: RawPulseDataset, resample: bool, shuffle: bool, workers: int
    ) -> DataLoader:
        # spawn, not fork: forking loader workers from a process that already
        # runs NCCL/CUDA threads can deadlock a DDP rank; spawn children start
        # clean, and persistent workers pay the startup cost once.
        return DataLoader(
            PretextDataset(
                raw, self.task, resample=resample, min_pulses=self.min_pulses
            ),
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
