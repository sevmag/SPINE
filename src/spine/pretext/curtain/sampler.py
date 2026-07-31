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
filtered out (the dataset then raises, failing loud). This is what lets
`PretextDataset.__getitem__` stay a pure index -> sample map with no
substitution.

Sensors are geometry-row indices throughout (position space); `pmt_id` is never
used here. Distances use raw metres via the k-NN graph in the geometry asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SamplerConfig:
    """Tunable knobs for the CURTAIN query sampler."""

    holdout_mode: str = "temporal"  # temporal | random
    q_lo: float = 0.3               # cutoff-time quantile drawn ~ U[q_lo, q_hi]
    q_hi: float = 0.7
    pos_k: int = 32                 # max positives per event (capped by supply)
    neg_anchor: str = "hidden"      # hidden (nearest-dark-to-positive) | visible-front
    rand_neg_frac: float = 0.15     # fraction of negatives drawn fully at random
    min_visible: int = 8            # min visible sensors for a valid split
    min_future: int = 4             # min future-new sensors for a valid split
    resample_tries: int = 6         # random cutoffs before the deterministic fallback


def load_geometry(path: str) -> Dict:
    """Load the geometry asset and attach a KDTree for pulse->sensor matching."""
    d = np.load(path)
    geo = {k: d[k] for k in d.files}
    geo["tree"] = cKDTree(geo["xyz"].astype(np.float64))
    return geo


def _sensor_index(px, py, pz, geo) -> np.ndarray:
    """Map each pulse to its geometry sensor index (positions are exact)."""
    dist, idx = geo["tree"].query(np.column_stack([px, py, pz]))
    assert dist.max() < 1e-2, f"pulse off-geometry by {dist.max():.3f} m"
    return idx.astype(np.int64)


def _nearest_dark(anchor: int, is_dark: np.ndarray, knn_idx: np.ndarray) -> int:
    """First dark sensor along `anchor`'s neighbour list, or -1 if none stored."""
    for nb in knn_idx[anchor]:
        if is_dark[nb]:
            return int(nb)
    return -1


def _temporal_split(pt, hit_sensors, first_t, cfg, rng):
    """Return (T, visible, future) for the temporal holdout, or None if the
    event is genuinely too sparse to split (< min_visible + min_future hits)."""
    for _ in range(cfg.resample_tries):
        Tc = float(np.quantile(pt, rng.uniform(cfg.q_lo, cfg.q_hi)))
        vis = hit_sensors[first_t[hit_sensors] < Tc]
        fut = hit_sensors[first_t[hit_sensors] >= Tc]
        if len(vis) >= cfg.min_visible and len(fut) >= cfg.min_future:
            return Tc, vis, fut
    # deterministic fallback: cut between the min_visible-th earliest and the
    # min_future-th latest hit sensor -- valid whenever there are enough hits.
    s = np.sort(first_t[hit_sensors])
    n = len(s)
    if n < cfg.min_visible + cfg.min_future:
        return None
    lo, hi = s[cfg.min_visible - 1], s[n - cfg.min_future]
    if hi <= lo:  # pathological time ties; real events do not hit this
        return None
    T = float((lo + hi) / 2.0)
    return T, hit_sensors[first_t[hit_sensors] < T], hit_sensors[first_t[hit_sensors] >= T]


def sample_event(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    pt: np.ndarray,
    pq: np.ndarray,
    geo: Dict,
    cfg: SamplerConfig,
    rng: np.random.Generator,
) -> Optional[Dict]:
    """Return the pretext split for one event, or None if too sparse to use.

    Output keys: `vis_pulse_mask [P]`, `query_idx [Q]`, `query_pos [Q,3]`,
    `query_label [Q]` (1=hit-after-T), `query_tag [Q]` ('pos'|'hard'|'rand'),
    `query_dt [Q]`, `T`, `t_cwm`. `t_cwm` is the charge-weighted mean time of the
    visible pulses -- a deterministic reference over exactly what the encoder
    sees, so the dt target carries no pretext randomness / future leakage.
    """
    n_sensors = geo["xyz"].shape[0]
    knn_idx = geo["knn_idx"]
    sensor = _sensor_index(px, py, pz, geo)

    hit_sensors = np.unique(sensor)
    first_t = np.full(n_sensors, np.inf)
    np.minimum.at(first_t, sensor, pt)
    is_dark = np.ones(n_sensors, dtype=bool)
    is_dark[hit_sensors] = False
    dark_pool = np.flatnonzero(is_dark)

    if cfg.holdout_mode == "temporal":
        split = _temporal_split(pt, hit_sensors, first_t, cfg, rng)
        if split is None:
            return None
        T, visible, future = split
        vis_pulse_mask = pt < T
    elif cfg.holdout_mode == "random":
        T = float("nan")
        perm = rng.permutation(hit_sensors)
        n_vis = int(round(0.5 * len(hit_sensors)))
        if n_vis < cfg.min_visible or len(hit_sensors) - n_vis < cfg.min_future:
            return None
        visible, future = perm[:n_vis], perm[n_vis:]
        vis_pulse_mask = np.isin(sensor, visible)
    else:
        raise ValueError(f"holdout_mode {cfg.holdout_mode!r} not implemented")

    if len(future) > cfg.pos_k:
        future = rng.choice(future, cfg.pos_k, replace=False)
    pos = future

    if cfg.neg_anchor == "visible-front":
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

    n_rand = int(round(cfg.rand_neg_frac * (len(hard) + 1)))
    avail = np.setdiff1d(dark_pool, hard, assume_unique=False)
    rand = (
        rng.choice(avail, min(n_rand, len(avail)), replace=False)
        if len(avail)
        else np.array([], dtype=np.int64)
    )

    neg = np.concatenate([hard, rand])
    query_idx = np.concatenate([pos, neg]).astype(np.int64)
    query_label = np.concatenate(
        [np.ones(len(pos)), np.zeros(len(neg))]
    ).astype(np.int64)
    query_tag = np.array(
        ["pos"] * len(pos) + ["hard"] * len(hard) + ["rand"] * len(rand)
    )
    w = np.clip(pq[vis_pulse_mask], 1e-2, None)
    t_cwm = float(np.sum(w * pt[vis_pulse_mask]) / np.sum(w))
    query_dt = np.where(
        query_label == 1, first_t[query_idx] - t_cwm, 0.0
    ).astype(np.float32)
    return {
        "vis_pulse_mask": vis_pulse_mask,
        "query_idx": query_idx,
        "query_pos": geo["xyz"][query_idx],
        "query_label": query_label,
        "query_tag": query_tag,
        "query_dt": query_dt,
        "T": T,
        "t_cwm": t_cwm,
        "n_visible": int(len(visible)),
        "n_future": int(len(future)),
        "n_dark": int(len(dark_pool)),
    }
