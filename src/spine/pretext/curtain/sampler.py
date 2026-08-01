"""CURTAIN query selection (pure numpy) -- the occupancy/light-front pretext.

One event -> (visible pulse mask, query positions, hit/no-hit labels, dt). A
randomized cutoff `T` (a quantile of the event's hit times) splits the event:
the encoder sees only pulses with t < T; positives are sensors whose *first* hit
is at t >= T (newly illuminated after the cut -- forecasting the light front,
not restating visible hits); negatives are dark sensors, the nearest dark DOM to
each positive plus an occasional random dark DOM.

A random cutoff in [q_lo, q_hi] can leave too few visible/future sensors even
for a good event (a per-epoch RNG artifact, not a property of the event). After
`resample_tries`, a DETERMINISTIC fallback picks a cutoff between the
min_visible-th earliest and min_future-th latest hit sensor, guaranteeing a
valid split for any event with >= min_visible + min_future hit sensors. So this
returns None only for genuinely too-sparse events -- which the caller must have
filtered out (`CurtainTask.make_sample` then raises, failing loud). This is what lets
`PretextDataset.__getitem__` stay a pure index -> sample map with no
substitution.

Sensors are geometry-row indices throughout, translated from the data-carried
sensor keys. Distances use raw metres via the k-NN graph in the geometry asset.
The sampler knobs are plain keyword arguments; their defaults live on
`CurtainTask`, the config-facing surface.
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
    holdout_mode: str,
    min_visible: int,
    min_future: int,
    random_vis_frac: float,
) -> bool:
    """Report whether sample_event is guaranteed to split this event.

    THE selection pre-filter predicate: events it accepts can never make
    `CurtainTask.make_sample` raise (for the same knob values), because it
    replicates the sampler's own guarantees -- the deterministic-fallback
    condition in temporal mode, the split arithmetic in random mode. Call it
    with the pulse times in the dtype the dataset feeds the sampler
    (float32): recomputing in a wider dtype disagrees at the margins, where
    near-tie first-hit times collapse under float32.

    Args:
        pt: Pulse times of the event.
        sensor_key: [P] integer sensor identity per pulse.
        holdout_mode: "temporal" (time cutoff) or "random" (sensor split).
        min_visible: Minimum visible sensors for a valid split.
        min_future: Minimum future-new sensors for a valid split.
        random_vis_frac: Visible fraction of the hit sensors; only consulted
            in "random" mode.

    Returns:
        True iff the event always yields a valid split.

    Raises:
        ValueError: On an unknown `holdout_mode`.
    """
    _, inverse = np.unique(sensor_key, return_inverse=True)
    n = int(inverse.max()) + 1 if len(sensor_key) else 0
    if n < min_visible + min_future:
        return False
    if holdout_mode == "temporal":
        first_t = np.full(n, np.inf)
        np.minimum.at(first_t, inverse, pt)
        s = np.sort(first_t)
        return bool(s[n - min_future] > s[min_visible - 1])
    if holdout_mode == "random":
        n_vis = int(round(random_vis_frac * n))
        return n_vis >= min_visible and (n - n_vis) >= min_future
    raise ValueError(f"holdout_mode {holdout_mode!r} not implemented")


def sample_event(
    pt: np.ndarray,
    pq: np.ndarray,
    geo: dict,
    rng: np.random.Generator,
    sensor: np.ndarray,
    *,
    holdout_mode: str,
    random_vis_frac: float,
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
        sensor: [P] geometry-row index per pulse, translated from the
            data's sensor keys (identity is never reconstructed from
            coordinates).
        holdout_mode: "temporal" (time cutoff) or "random" (sensor split).
        random_vis_frac: Visible fraction of the hit sensors; only consulted
            in "random" mode.
        q_lo: Lower bound of the cutoff-quantile window.
        q_hi: Upper bound of the cutoff-quantile window.
        pos_k: Maximum positives per event (capped by supply).
        neg_anchor: "hidden" (nearest dark to each positive) or
            "visible-front" (nearest dark to the latest visible sensors).
        rand_neg_frac: Fraction of negatives drawn fully at random.
        min_visible: Minimum visible sensors for a valid split.
        min_future: Minimum future-new sensors for a valid split.
        resample_tries: Random cutoffs before the deterministic fallback.

    Returns:
        None if the event is too sparse to use, else a dict with
        `vis_pulse_mask [P]`, `query_pos [Q,3]`, `query_label [Q]`
        (1=hit-after-T), `query_hard [Q]` (True for positives and
        nearest-dark negatives, False for random negatives), `query_dt [Q]`
        and `t_cwm` -- the charge-weighted mean time of the visible pulses, a
        deterministic reference over exactly what the encoder sees, so the dt
        target carries no pretext randomness / future leakage.

    Raises:
        ValueError: On an unknown `holdout_mode`.
    """
    n_sensors = geo["xyz"].shape[0]
    knn_idx = geo["knn_idx"]
    hit_sensors = np.unique(sensor)
    first_t = np.full(n_sensors, np.inf)
    np.minimum.at(first_t, sensor, pt)
    is_dark = np.ones(n_sensors, dtype=bool)
    is_dark[hit_sensors] = False
    dark_pool = np.flatnonzero(is_dark)

    if holdout_mode == "temporal":
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
    elif holdout_mode == "random":
        T = float("nan")
        perm = rng.permutation(hit_sensors)
        n_vis = int(round(random_vis_frac * len(hit_sensors)))
        if n_vis < min_visible or len(hit_sensors) - n_vis < min_future:
            return None
        visible, future = perm[:n_vis], perm[n_vis:]
        vis_pulse_mask = np.isin(sensor, visible)
    else:
        raise ValueError(f"holdout_mode {holdout_mode!r} not implemented")

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
