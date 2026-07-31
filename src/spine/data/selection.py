"""Frozen train/val/test split -- the reproducibility contract.

The eval set is a fixed front-slice of an ordered selection; the shared
pretrain/finetune pool is everything after. Boundaries are module constants (not
run knobs) so nothing can silently shift them, and eval is disjoint from the
pool by construction. A leak here silently
inflates every pretrained-vs-scratch comparison.

Also the home of `filter_by_min_count`: the caller pre-filters the pool to
usable events (>= min_visible + min_future hit sensors) so `__getitem__` can
stay a pure index -> sample map and fail loud on anything unusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pyarrow.parquet as pq

N_TEST = 49_920
VAL_START = 50_000
N_VAL = 20_000
POOL_START = VAL_START + N_VAL  # 70_000


@dataclass
class Split:
    """A frozen (test, val, pool) split of event ids."""

    test: np.ndarray
    val: np.ndarray
    pool: np.ndarray  # shared pretrain/finetune pool; train = pool[:n_train]


def load_event_nos(parquet_path: str, column: str = "event_no") -> np.ndarray:
    """Ordered event ids from a selection parquet (order defines the split)."""
    return pq.read_table(parquet_path, columns=[column])[column].to_numpy()


def make_split(events: np.ndarray) -> Split:
    """Slice the frozen [test | val | pool] from an ordered selection."""
    if len(events) <= POOL_START:
        raise ValueError(f"selection has {len(events)} events; need > {POOL_START}")
    return Split(
        test=events[:N_TEST],
        val=events[VAL_START : VAL_START + N_VAL],
        pool=events[POOL_START:],
    )


def train_events(split: Split, n_train: int) -> np.ndarray:
    """First n_train pool events (nested: 10k train is a prefix of 100k)."""
    if n_train > len(split.pool):
        raise ValueError(f"pool has {len(split.pool)} events; asked {n_train}")
    return split.pool[:n_train]


def assert_disjoint(split: Split, n_pool_check: Optional[int] = None) -> None:
    """Fail loud if eval leaks into the pool or has duplicate ids.

    `n_pool_check` bounds how far into the pool to check (the full pool can be
    tens of millions); the frozen 1.1M window is the relevant coverage.
    """
    test = set(split.test.tolist())
    val = set(split.val.tolist())
    pool_slice = split.pool[:n_pool_check] if n_pool_check else split.pool
    pool = set(pool_slice.tolist())
    assert len(test) == N_TEST, "duplicate event_no inside the test front"
    assert len(val) == N_VAL, "duplicate event_no inside the val front"
    assert not (test & pool), "test leaks into the pretrain/finetune pool"
    assert not (val & pool), "val leaks into the pretrain/finetune pool"
    assert not (test & val), "test and val overlap"


def filter_by_min_count(events: np.ndarray, counts: Mapping[int, int],
                        min_count: int) -> np.ndarray:
    """Keep events whose precomputed count >= min_count (caller's pre-filter).

    `counts` maps event_no -> count, precomputed offline (e.g. number of hit
    sensors). The CURTAIN sampler needs >= min_visible + min_future hit sensors
    to guarantee a split, so filter on hit-sensor count (a pulse count is a
    looser proxy). The dataset assumes the selection is already filtered and
    raises otherwise -- this is how you produce a clean selection.
    """
    if len(events) == 0:
        return events
    keep = np.array([counts.get(int(e), 0) >= min_count for e in events], bool)
    return events[keep]
