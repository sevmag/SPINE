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
| `data/` | geometry, FeatureScaler scaling, datamodule (reader- & selection-agnostic) | scaling + datamodule real |
| `backbones/` | encoder interface (swappable; graphnet-free) | interface real; DeepIce impl in examples/, wired to collate |
| `pretext/` | pretext-task interface + `curtain/` | interface + curtain sampler/head/objectives/task real |
| `engine.py` | Lightning SSLModule + optim/sched (transfer export in `utils.py`) | real |
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
- **Core has zero graphnet dependency.** `src/spine/` imports no graphnet; the
  reference DeepIce backbone and the readers live in `examples/` as the graphnet
  integration. The backbone drives `DeepIce`'s submodules directly over the
  collated jagged nested tensor (`to_padded_tensor` in the backbone) — no
  `array_to_sequence`, no torch_geometric — so the `Backbone` contract is a plain
  nested tensor, model-agnostic (padding or varlen backbones both project from
  it). The finetuning bench loads
  `ckpt["backbone"]` into graphnet DeepIce, so the exported state_dict must stay
  compatible — keep DeepIce, or vendor a state-dict-identical encoder later
  (`examples/deepice_backbone.py` TODO).
- **Data layer.** Pretext needs **raw** pulses (the Δt reference is
  charge-weighted-mean-time on raw values), so standardization runs at the model
  boundary **after** the split, not in the source. LMDB is welcome for speed but
  as the **low-level read utilities** behind the read `Dataset` (raw pulses; identity
  detector; no truth/labels) — not the full `LMDBDataset` (which drags the graph
  pipeline back into the data layer). SPINE ships **no reader** -- it consumes any Dataset meeting the RawPulseDataset contract (`raw[i] -> {event_no, pulses[P,5] raw}`); graphnet's LMDBDataset/SQLiteDataset are the recommended readers (adapter: examples/readers.py). Profile the loader before converting.
- **Batch = jagged nested tensor.** `collate` emits the pulses and the query
  fields as `torch.nested` jagged NJTs (no padding baked in); each backbone
  projects — DeepIce calls `to_padded_tensor`, a varlen backbone would read
  `offsets`. Verified to survive the multiprocessing DataLoader and Lightning's
  GPU batch transfer (pin_memory + `move_data_to_device`) on torch 2.6. This is
  NJT-as-**transport** only: NJT-in-attention stays off-limits on this torch (the
  rel-spacetime kernel bugs).
- **Selections are the caller's.** SPINE owns no split: `fit` takes train/val
  readers, and the caller supplies disjoint train/val event lists (and keeps the
  test set off-limits). A leak inflates every pretrained-vs-scratch number, so
  that hygiene lives with whoever builds the selection (e.g. om_adapter_bench's
  frozen split), not in this package.
- **DDP correctness baked in.** `sync_dist=True` on the val metric (else
  ReduceLROnPlateau desyncs replica LRs); the datamodule is **fail-loud** —
  it pre-filters and make_sample is guaranteed to split any surviving event, so
  it never silently drops events and a batch is never empty (empty batch → a rank
  skips its step → NCCL deadlock); and runs should set `TORCH_NCCL_ENABLE_MONITORING=0` (the
  heartbeat-monitor false positive that killed the 5M/1M runs).

## Transfer contract (the boundary artifact)
`utils.py`'s `TransferCheckpoint` writes `{backbone, full_state, config, step, val_loss}` on
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
3. Config system (hydra), profile loader.
Deferred: other backbones, other pretexts, multi-detector, in-repo eval.

## Open decisions
1. Config: **hydra** (matches NuBench stack) vs dataclass+CLI.
2. graphnet DeepIce behind the interface vs **vendor** a standalone encoder.
3. Repo scope: pretraining-only (recommended) vs pull in a finetuning harness.
4. Multi-detector: single-detector MVP (geometry-parametrized) vs multi from day one.
