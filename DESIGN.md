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
| block | responsibility |
|---|---|
| `data/` | geometry asset + sensor-key lookup, FeatureScaler scaling, datamodule (reader- & selection-agnostic) |
| `backbones/` | encoder interface (swappable; graphnet-free; DeepIce impl in integrations/spine_graphnet/) |
| `pretext/` | pretext-task interface + `curtain/` (sampler, head, objectives, task, val callbacks) |
| `ssl_module.py` | Lightning module; optimizer/scheduler injected as factories (transfer export in `utils.py`) |
| `configs/` + `train.py` | Hydra groups compose a run (examples/train_curtain.py); fit() assembles |

## The two interfaces (all extensibility lives here)
- **`Backbone.encode(batch) -> EncodedEvent(tokens, token_mask, cls)`** — swap
  architectures without touching pretext/engine.
- **`PretextTask`** — `make_sample` (CPU: mask/target), `collate`, `build_head`,
  `loss`. A task carries a list of weighted **`Objective`s**, each an abstract
  class owning its own head (`build_head`) and `loss`, over one sample.

### v1 vs v2 is an objective, not a fork
`sampler.py` computes `query_dt` for every event; v1 scores
`[OccupancyObjective]`, v2 adds `DtObjective(weight)`. Each objective owns its head (off a shared
per-query embedding) and its loss + masking, so v2 is `task/objectives=v2`
— no forked
model/dataset/train files.

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
  pipeline back into the data layer). SPINE ships **no reader** — it consumes any
  Dataset meeting the RawPulseDataset contract (canonical statement:
  `spine/data/datamodule.py`), and readers must emit per-pulse `sensor_key`
  identity. Profile the loader before converting.
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
  that hygiene lives with whoever builds the selection — filtered with
  `sampler.can_always_split`, the predicate stating exactly which events the
  sampler is guaranteed to handle.
- **Sensor identity is data-carried.** Readers emit an integer `sensor_key`
  per pulse (multi-level string/module/PMT IDs composed by the reader;
  single-PMT detectors use 1 for the missing level);
  `load_geometry(sensor_key=...)` resolves keys to rows through a per-row
  array verified offline at asset-build time. Reconstructing identity from
  coordinates is not supported: float matching collapses near-duplicate
  positions and near-tie times at the margins, exactly where training then
  crashes.
- **Transfer invariant: extend the encoder, don't wrap it.** The checkpoint's
  `backbone` entry is the backbone's own state_dict and must load directly
  into the downstream encoder — a wrapper would prefix every key and silently
  break that load. Hence `DeepIceBackbone(DeepIce, Backbone)` subclasses the
  encoder.
- **Epoch-global metrics are callbacks.** Rank-based metrics (the occupancy
  AUCs) cannot flow through per-batch log averaging; callbacks cache per
  batch and reduce once per epoch, keeping task and engine interfaces closed.
- **Optimizer/scheduler are injected factories.** `SSLModule` takes
  `params -> Optimizer` / `Optimizer -> scheduler` callables (Hydra
  `_partial_` configs); swapping optimization is a config override, not a
  code change.
- **DDP correctness baked in.** `sync_dist=True` on the val metric (else
  ReduceLROnPlateau desyncs replica LRs); the data path is **fail-loud** —
  the caller pre-filters and `make_sample` raises on any event it cannot split
  (never returns None, never skips), so a batch is never empty (empty batch → a
  rank skips its step → NCCL deadlock); and multi-GPU runs should set
  `TORCH_NCCL_ENABLE_MONITORING=0` — PyTorch's NCCL heartbeat monitor can
  abort a healthy run on a GIL-starved false positive during heavy compute.

## Transfer contract (the boundary artifact)
`utils.py`'s `TransferCheckpoint` writes `{backbone, full_state, config, step, val_loss}` on
best val (rank-0 only). Downstream loads `ckpt["backbone"]`. Finetuning/eval
stays in the existing bench — this repo emits encoders, nothing more.

## Adding a method (extensibility test)
New folder under `pretext/`, point a `task/<name>.yaml` `_target_` at the
new `PretextTask`:
- **MAE**: `make_sample` masks pulses; head = decoder; loss = reconstruct.
- **Contrastive**: `make_sample` = two views; head = projection on `cls`; loss = NT-Xent.
Data, backbone, engine unchanged.

## MVP order
1. Wire end-to-end and reproduce v1 (occupancy) ✓ — validated against the
   reference pretraining (best val loss within noise, AUCs within 7e-4).
2. Reproduce v2 by config (`task/objectives=v2` exists; revalidation open).
3. Config system (hydra) ✓ — profile loader remains.
Deferred: other backbones, other pretexts, multi-detector, in-repo eval.

## Open decisions
1. graphnet DeepIce behind the interface vs **vendor** a standalone encoder.
2. Repo scope: pretraining-only (recommended) vs pull in a finetuning harness.
3. Multi-detector: single-detector MVP (geometry-parametrized) vs multi from day one.
