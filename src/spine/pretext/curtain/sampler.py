"""CURTAIN query selection (pure numpy) -- the occupancy/light-front pretext.

One event -> (visible pulse mask, query positions, hit/no-hit labels, dt). A
randomized cutoff `T` (a quantile of the event's hit times) splits the event:
the encoder sees only pulses with t < T; positives are sensors whose *first* hit
is at t >= T (newly illuminated after the cut -- forecasting the light front,
not restating visible hits); negatives are dark sensors, the nearest dark DOM to
each positive plus an occasional random dark DOM.

Sensors are geometry-row indices throughout (position space); `pmt_id` is never
used here. Distances use raw metres via the k-NN graph in the geometry asset.
Ported verbatim from the occupancy study so the pretext lives in one place.
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
    min_visible: int = 8            # min visible sensors, else resample/skip
    min_future: int = 4             # min future-new sensors, else resample/skip
    resample_tries: int = 6


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
    """Return the pretext split for one event, or None if it cannot be used.

    Output keys: `vis_pulse_mask [P]` (encoder input pulses), `query_idx [Q]`
    (sensor indices), `query_pos [Q,3]`, `query_label [Q]` (1=hit-after-T),
    `query_tag [Q]` ('pos'|'hard'|'rand'), `query_dt [Q]`, `T`, `t_cwm`.

    `t_cwm` is the charge-weighted mean time of the visible pulses -- the event's
    time reference. Unlike the cutoff `T` (a randomized sampling artifact), it is
    a deterministic function of exactly what the encoder sees, so the dt target
    carries no pretext randomness and no future leakage.
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
        for _ in range(cfg.resample_tries):
            q = rng.uniform(cfg.q_lo, cfg.q_hi)
            T = float(np.quantile(pt, q))
            visible = hit_sensors[first_t[hit_sensors] < T]
            future = hit_sensors[first_t[hit_sensors] >= T]
            if len(visible) >= cfg.min_visible and len(future) >= cfg.min_future:
                break
        else:
            return None
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
