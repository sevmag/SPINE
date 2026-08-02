"""Build pretraining train/val selection parquets from a subset DB.

CC event_nos in DB order, minus the held-out front of the ordered selection,
deterministically shuffled; val = the first `n_val_events`, train = the next
`n_train`. Both slices are filtered with `sampler.can_always_split` on
(float32 times, sensor keys), so the fail-loud runtime can never reject a
selected event. Order is preserved: a smaller run's train list is a prefix of
a larger one's (kept counts at each --slice-mark are printed for n_train).

Run on a compute node (sbatch/srun) -- it scans pulses for every candidate.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from multiprocessing import Pool

import numpy as np
import pandas as pd

from spine.pretext.curtain.sampler import can_always_split

_DB = None
_ARGS = None


def _init(db: str, min_visible: int, min_future: int) -> None:
    global _DB, _ARGS
    _DB = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    _ARGS = (min_visible, min_future)


def _keep(ev: int) -> bool:
    min_visible, min_future = _ARGS
    rows = _DB.execute(
        "SELECT t, pmt_id FROM merged_photons WHERE event_no=?", (int(ev),)
    ).fetchall()
    if len(rows) < min_visible + min_future:
        return False
    a = np.asarray(rows)
    return can_always_split(
        a[:, 0].astype(np.float32),
        a[:, 1].astype(np.int64),
        holdout_mode="temporal",
        min_visible=min_visible,
        min_future=min_future,
        random_vis_frac=0.5,
    )


def _filter(events: np.ndarray, args: argparse.Namespace, tag: str) -> np.ndarray:
    keep: list[bool] = []
    with Pool(
        args.workers,
        initializer=_init,
        initargs=(args.db, args.min_visible, args.min_future),
    ) as pool:
        for i, k in enumerate(pool.imap(_keep, events, chunksize=64)):
            keep.append(k)
            if (i + 1) % 50_000 == 0:
                print(
                    f"  {tag}: {i + 1}/{len(events)} scanned, kept {int(np.sum(keep))}",
                    flush=True,
                )
    return np.asarray(keep, bool)


def main() -> None:
    """Build, filter and write the selection parquets."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument(
        "--heldout-parquet",
        required=True,
        help="ordered selection whose front is the eval holdout",
    )
    p.add_argument("--n-heldout", type=int, default=70_000)
    p.add_argument("--n-val-events", type=int, default=4_000)
    p.add_argument("--n-train", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-visible", type=int, default=8)
    p.add_argument("--min-future", type=int, default=4)
    p.add_argument("--expect-pool", type=int, default=0)
    p.add_argument("--slice-marks", type=int, nargs="*", default=[100_000, 500_000])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    ev = np.array(
        [
            r[0]
            for r in con.execute("SELECT event_no FROM mc_truth WHERE interaction=1")
        ],
        dtype=np.int64,
    )
    con.close()
    heldout = set(
        pd.read_parquet(args.heldout_parquet)["event_no"]
        .to_numpy()[: args.n_heldout]
        .tolist()
    )
    pool_ev = np.array([e for e in ev if e not in heldout], dtype=np.int64)
    print(f"pool={len(pool_ev)} (cc={len(ev)})", flush=True)
    if args.expect_pool and len(pool_ev) != args.expect_pool:
        raise SystemExit(f"pool {len(pool_ev)} != expected {args.expect_pool}")

    np.random.default_rng(args.seed).shuffle(pool_ev)
    val = pool_ev[: args.n_val_events]
    train = pool_ev[args.n_val_events : args.n_val_events + args.n_train]

    vk = _filter(val, args, "val")
    print(f"val: kept {int(vk.sum())}/{len(val)}", flush=True)
    tk = _filter(train, args, "train")
    print(f"train: kept {int(tk.sum())}/{len(train)}", flush=True)
    for m in args.slice_marks:
        print(f"  kept within train[:{m}] = {int(tk[:m].sum())}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, arr in (("val", val[vk]), ("train", train[tk])):
        out = os.path.join(args.out_dir, f"pretrain_{name}.parquet")
        pd.DataFrame({"event_no": arr}).to_parquet(out, index=False)
        print(f"wrote {out} ({len(arr)} events)", flush=True)


if __name__ == "__main__":
    main()
