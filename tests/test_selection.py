"""The frozen split must stay leak-free -- the one test that can't regress."""

import numpy as np
import pytest

from spine.data import selection as sel


def _events(n):
    # unique, shuffled ids: order defines the split, uniqueness guards leakage
    rng = np.random.default_rng(0)
    return rng.permutation(n).astype(np.int64)


def test_split_sizes_and_disjoint():
    ev = _events(200_000)
    s = sel.make_split(ev)
    assert len(s.test) == sel.N_TEST
    assert len(s.val) == sel.N_VAL
    sel.assert_disjoint(s)


def test_train_is_nested_prefix():
    ev = _events(200_000)
    s = sel.make_split(ev)
    a = sel.train_events(s, 10_000)
    b = sel.train_events(s, 100_000)
    assert np.array_equal(a, b[:10_000])


def test_duplicate_in_eval_front_is_caught():
    ev = _events(200_000)
    ev[5] = ev[10]  # inject a duplicate inside the test front
    s = sel.make_split(ev)
    with pytest.raises(AssertionError):
        sel.assert_disjoint(s)


def test_too_small_selection_errors():
    with pytest.raises(ValueError):
        sel.make_split(_events(sel.POOL_START))
