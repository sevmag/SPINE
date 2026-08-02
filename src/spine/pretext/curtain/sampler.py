"""CURTAIN query sampling (pure numpy).

A cutoff splits each event: the encoder sees the pulses before it; queries
are the sensors newly lit after it (positives) and dark sensors (nearest-dark
per positive + random, the negatives). After `resample_tries` random cutoffs
a deterministic fallback guarantees a split exactly when `can_always_split`
holds. Sensor identity arrives as geometry-row indices; knob defaults live on
CurtainTask.
"""

from __future__ import annotations

import numpy as np


def _nearest_dark(anchor: int, is_dark: np.ndarray, knn_idx: np.ndarray) -> int:
    """Find the first dark sensor along a neighbour list.

    Args:
        anchor: Geometry index whose neighbour list is walked.
        is_dark: [n_sensors] mask, True where the sensor has no hit.
        knn_idx: [n_sensors, k] precomputed nearest-neighbour indices.

    Returns:
        The first dark neighbour's index, or -1 if none is stored.
    """
    for nb in knn_idx[anchor]:
        if is_dark[nb]:
            return int(nb)
    return -1


def _temporal_split(
    pt: np.ndarray,
    hit_sensors: np.ndarray,
    first_t: np.ndarray,
    rng: np.random.Generator,
    *,
    q_lo: float,
    q_hi: float,
    min_visible: int,
    min_future: int,
    resample_tries: int,
):
    """Choose the temporal cutoff and split the hit sensors around it.

    Args:
        pt: Pulse times of the event.
        hit_sensors: Geometry indices with at least one pulse.
        first_t: [n_sensors] first-hit time per sensor (inf where dark).
        rng: Generator for the random cutoff draws.
        q_lo: Lower bound of the cutoff-quantile window.
        q_hi: Upper bound of the cutoff-quantile window.
        min_visible: Minimum visible sensors for a valid split.
        min_future: Minimum future-new sensors for a valid split.
        resample_tries: Random cutoffs before the deterministic fallback.

    Returns:
        (T, visible, future), or None when the event is genuinely too sparse
        to split (< min_visible + min_future hit sensors).
    """
    for _ in range(resample_tries):
        Tc = float(np.quantile(pt, rng.uniform(q_lo, q_hi)))
        vis = hit_sensors[first_t[hit_sensors] < Tc]
        fut = hit_sensors[first_t[hit_sensors] >= Tc]
        if len(vis) >= min_visible and len(fut) >= min_future:
            return Tc, vis, fut
    # deterministic fallback: cut between the min_visible-th earliest and the
    # min_future-th latest hit sensor -- valid whenever there are enough hits.
    s = np.sort(first_t[hit_sensors])
    n = len(s)
    if n < min_visible + min_future:
        return None
    lo, hi = s[min_visible - 1], s[n - min_future]
    if hi <= lo:  # pathological time ties; real events do not hit this
        return None
    T = float((lo + hi) / 2.0)
    return (
        T,
        hit_sensors[first_t[hit_sensors] < T],
        hit_sensors[first_t[hit_sensors] >= T],
    )


def can_always_split(
    pt: np.ndarray,
    sensor_key: np.ndarray,
    *,
    min_visible: int,
    min_future: int,
) -> bool:
    """Report whether sample_event is guaranteed to split this event.

    THE selection pre-filter predicate: accepted events can never make
    make_sample raise (same knob values) -- it replicates the sampler's own
    guarantees. Pass times as float32 (the dtype the sampler sees); wider
    dtypes disagree at near-tie margins.

    Args:
        pt: Pulse times of the event.
        sensor_key: [P] integer sensor identity per pulse.
        min_visible: Minimum visible sensors for a valid split.
        min_future: Minimum future-new sensors for a valid split.

    Returns:
        True iff the event always yields a valid split.
    """
    _, inverse = np.unique(sensor_key, return_inverse=True)
    n = int(inverse.max()) + 1 if len(sensor_key) else 0
    if n < min_visible + min_future:
        return False
    first_t = np.full(n, np.inf)
    np.minimum.at(first_t, inverse, pt)
    s = np.sort(first_t)
    return bool(s[n - min_future] > s[min_visible - 1])


def sample_event(
    pt: np.ndarray,
    pq: np.ndarray,
    geo: dict,
    rng: np.random.Generator,
    sensor: np.ndarray,
    *,
    q_lo: float,
    q_hi: float,
    pos_k: int,
    neg_anchor: str,
    rand_neg_frac: float,
    min_visible: int,
    min_future: int,
    resample_tries: int,
) -> dict | None:
    """Build the pretext split for one event.

    Args:
        pt: Pulse times.
        pq: Pulse charges.
        geo: Geometry asset (xyz, knn_idx).
        rng: Generator; the cutoff and negatives re-randomize per call.
        sensor: [P] geometry-row index per pulse (from the data's keys).
        q_lo: Lower bound of the cutoff-quantile window.
        q_hi: Upper bound of the cutoff-quantile window.
        pos_k: Maximum positives per event (capped by supply).
        neg_anchor: "hidden" (nearest dark per positive) or "visible-front".
        rand_neg_frac: Fraction of negatives drawn fully at random.
        min_visible: Minimum visible sensors for a valid split.
        min_future: Minimum future-new sensors for a valid split.
        resample_tries: Random cutoffs before the deterministic fallback.

    Returns:
        None if too sparse, else `vis_pulse_mask [P]`, `query_pos [Q,3]`,
        `query_label [Q]`, `query_hard [Q]` (False = random negative),
        `query_dt [Q]` and `t_cwm` (charge-weighted mean visible time -- a
        deterministic dt reference with no future leakage).
    """
    n_sensors = geo["xyz"].shape[0]
    knn_idx = geo["knn_idx"]
    hit_sensors = np.unique(sensor)
    first_t = np.full(n_sensors, np.inf)
    np.minimum.at(first_t, sensor, pt)
    is_dark = np.ones(n_sensors, dtype=bool)
    is_dark[hit_sensors] = False
    dark_pool = np.flatnonzero(is_dark)

    split = _temporal_split(
        pt,
        hit_sensors,
        first_t,
        rng,
        q_lo=q_lo,
        q_hi=q_hi,
        min_visible=min_visible,
        min_future=min_future,
        resample_tries=resample_tries,
    )
    if split is None:
        return None
    T, visible, future = split
    vis_pulse_mask = pt < T

    if len(future) > pos_k:
        future = rng.choice(future, pos_k, replace=False)
    pos = future

    if neg_anchor == "visible-front":
        order = visible[np.argsort(-first_t[visible])]
        anchors = order[: len(pos)]
    else:
        anchors = pos
    hard = []
    used = set()
    for a in anchors:
        nb = _nearest_dark(int(a), is_dark, knn_idx)
        if nb >= 0 and nb not in used:
            used.add(nb)
            hard.append(nb)
    hard = np.array(hard, dtype=np.int64)

    n_rand = int(round(rand_neg_frac * (len(hard) + 1)))
    avail = np.setdiff1d(dark_pool, hard, assume_unique=False)
    rand = (
        rng.choice(avail, min(n_rand, len(avail)), replace=False)
        if len(avail)
        else np.array([], dtype=np.int64)
    )

    neg = np.concatenate([hard, rand])
    query_idx = np.concatenate([pos, neg]).astype(np.int64)
    query_label = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(
        np.int64
    )
    query_hard = np.concatenate(
        [np.ones(len(pos) + len(hard), bool), np.zeros(len(rand), bool)]
    )
    w = np.clip(pq[vis_pulse_mask], 1e-2, None)
    t_cwm = float(np.sum(w * pt[vis_pulse_mask]) / np.sum(w))
    query_dt = np.where(query_label == 1, first_t[query_idx] - t_cwm, 0.0).astype(
        np.float32
    )
    return {
        "vis_pulse_mask": vis_pulse_mask,
        "query_pos": geo["xyz"][query_idx],
        "query_label": query_label,
        "query_hard": query_hard,
        "query_dt": query_dt,
        "t_cwm": t_cwm,
    }
