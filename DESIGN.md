# SPINE design

Self-supervised Pretraining In Neutrino Experiments. The repo produces
**pretrained backbones** (encoder checkpoints) that downstream supervised
benchmarks fine-tune. First pretext: **CURTAIN** (occupancy / light-front
forecast); built so new SSL methods are small plugins.

## Mental model
A run is: **Data → Backbone → Pretext(head + targets + loss)**, wired by an
**Engine**, named by a **Config**. The only thing you write to add a method is a
`pretext/` plugin — data, backbone, and engine are reused unchanged.

## Blocks
| block | responsibility | status |
|---|---|---|
| `data/` | event sources (sqlite/lmdb), frozen split, geometry, standardize | selection + standardize + sqlite real; lmdb + datamodule real skeleton |
| `backbones/` | encoder interface + DeepIce wrapper (swappable) | interface real; deepice token-forward ported |
| `pretext/` | pretext-task interface + `curtain/` | interface + curtain sampler/head/objectives/task real |
| `engine/` | Lightning module, optim/sched, transfer-checkpoint export | real |
| `configs/`, `train.py` | compose + fit | argparse skeleton (hydra TODO) |

## The two interfaces (all extensibility lives here)
- **`Backbone.encode(batch) -> EncodedEvent(tokens, token_mask, cls)`** — swap
  architectures without touching pretext/engine.
- **`PretextTask`** — `make_sample` (CPU: mask/target), `collate`, `build_head`,
  `loss`. A task carries a list of weighted **`Objective`s** over one sample.

### v1 vs v2 is an objective, not a fork
`sampler.py` computes `query_dt` for every event; v1 scores `[OCCUPANCY]`, v2
scores `[OCCUPANCY, dt(λ)]`. The head width and the loss terms come from the
objective list, so v2 is `--task-objectives occupancy,dt` — no forked
model/dataset/train files (the occupancy study had three).

## Decisions
- **GraphNeT surface = just the encoder.** Only `DeepIce` (+ `array_to_sequence`)
  is used, via internals, behind `Backbone`. Everything else here is ours. The
  finetuning bench loads `ckpt["backbone"]` into graphnet DeepIce, so the
  exported state_dict must stay compatible — keep DeepIce, or vendor a
  state-dict-identical encoder later (`backbones/deepice.py` TODO).
- **Data layer.** Pretext needs **raw** pulses (the Δt reference is
  charge-weighted-mean-time on raw values), so standardization runs at the model
  boundary **after** the split, not in the source. LMDB is welcome for speed but
  as the **low-level read utilities** behind `EventSource` (raw pulses; identity
  detector; no truth/labels) — not the full `LMDBDataset` (which drags the graph
  pipeline back into the data layer). Profile the loader before converting.
- **Frozen split is tested (`tests/test_selection.py`).** Boundaries are
  constants; eval is disjoint from the pool by construction. A leak here
  silently inflates every pretrained-vs-scratch number.
- **DDP correctness baked in.** `sync_dist=True` on the val metric (else
  ReduceLROnPlateau desyncs replica LRs); the datamodule **substitutes unusable
  events** so a batch is never empty (empty batch → a rank skips its step →
  NCCL deadlock); and runs should set `TORCH_NCCL_ENABLE_MONITORING=0` (the
  heartbeat-monitor false positive that killed the 5M/1M runs).

## Transfer contract (the boundary artifact)
`engine/transfer.py` writes `{backbone, full_state, config, step, val_loss}` on
best val (rank-0 only). Downstream loads `ckpt["backbone"]`. Finetuning/eval
stays in the existing bench — this repo emits encoders, nothing more.

## Adding a method (extensibility test)
New folder under `pretext/`, register a `PretextTask`:
- **MAE**: `make_sample` masks pulses; head = decoder; loss = reconstruct.
- **Contrastive**: `make_sample` = two views; head = projection on `cls`; loss = NT-Xent.
Data, backbone, engine unchanged.

## MVP order
1. Wire end-to-end and reproduce v1 (occupancy) → checkpoint-compatible with the bench.  ← current scaffold
2. Add `dt` objective → reproduce v2 by config.
3. Config system (hydra), profile loader, tests.
Deferred: other backbones, other pretexts, multi-detector, in-repo eval.

## Open decisions
1. Config: **hydra** (matches NuBench stack) vs dataclass+CLI.
2. graphnet DeepIce behind the interface vs **vendor** a standalone encoder.
3. Repo scope: pretraining-only (recommended) vs pull in a finetuning harness.
4. Multi-detector: single-detector MVP (geometry-parametrized) vs multi from day one.
