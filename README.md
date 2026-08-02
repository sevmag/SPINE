# SPINE: Self-supervised Pretraining In Neutrino Experiments

Extensible self-supervised pretraining of transformers on neutrino-telescope
data. The repo produces **pretrained backbones** (encoder checkpoints) that
downstream supervised benchmarks load and fine-tune. A *spine* is a backbone,
which is exactly what this emits.

The first pretext task is **CURTAIN** (occupancy / light-front forecast). The
architecture is built so a new self-supervised method is a small plugin under
`pretext/`, reusing the data, backbone, and training engine unchanged.

## 🧭 Philosophy
The core is framework-agnostic and fits neatly into plain PyTorch: it depends
only on torch, pytorch-lightning and numpy. Readers are ordinary indexable
`Dataset`s emitting a small canonical sample format, models are `nn.Module`s
behind two narrow interfaces (`Backbone`, `PretextTask`), and `fit()` takes
injected factories and callbacks. Hydra and graphnet integrate neatly, but both are
strictly optional conveniences: use either, both, or neither. Around that
core you choose your frame:

- **Bring your own.** Wire SPINE into your existing code for data loading,
  configuration, versioning and logging; nothing in the core assumes Hydra,
  wandb or graphnet.
- **Use the Hydra configs.** The `configs/` tree composes complete runs
  (backbone, task, optimizer, scheduler, callbacks, trainer, data) through
  the launcher in `examples/`; every component is swapped by a config
  override instead of a code change, with or without graphnet.
- **Use the graphnet frame.** `integrations/spine_graphnet` provides the DeepIce backbone
  and a reader adapter over graphnet's datasets, and the emitted encoders
  load straight into graphnet's benchmarks.

## 🗂️ Layout
```
src/spine/
  data/       geometry + FeatureScaler scaling, datamodule (reader- & selection-agnostic)
  backbones/  encoder interface (swappable; graphnet-free core)
  pretext/    pretext-task interface + curtain/ (the first task)
  ssl_module.py  Lightning SSLModule (optimizer/scheduler injected as factories)
  utils.py    TransferCheckpoint callback (best-val backbone export)
  train.py    reader-agnostic fit() assembly
configs/      Hydra groups: backbone/ task/ optimizer/ scheduler/ callbacks/ trainer/ data/
integrations/spine_graphnet/  graphnet integration: DeepIce backbone + reader adapter
examples/     Hydra launcher (train_curtain.py) + runnable graphnet demo (graphnet_demo.py)
tests/        core-independence gate (spine imports no graphnet)
```

See `DESIGN.md` for the module decomposition, interfaces, and the decisions
behind them (graphnet surface, LMDB, selections stay the caller's, DDP gotchas).

## ⚙️ Runtime
Needs Python 3.10, torch 2.6 and pytorch-lightning >= 2.5 (see
`pyproject.toml`); install with `pip install -e .`, which provides `spine`
and `spine_graphnet`. The DeepIce backbone additionally needs a graphnet
checkout on the import path (graphnet is not on PyPI). Pretraining and
finetuning must share one environment: the emitted encoders are loaded back
into graphnet's DeepIce downstream.

## 🚀 Running
Runs are composed with **Hydra** from `configs/` and launched via
`examples/train_curtain.py`; the core library is Hydra-free (every component is
`instantiate`d from config at the launcher). Override any group or value:
```
python examples/train_curtain.py \
    geo=geometry.npz geo_sensor_key=pmt_id out=ckpt.pth data.db=hexagon.db \
    data.train_selection=train.parquet data.val_selection=val.parquet \
    callbacks=curtain_auc task/objectives=v2 trainer.devices=4
```

An end-to-end demo of the graphnet frame needs no data of your own: it runs on
the Prometheus example file bundled with a graphnet checkout, pretrains a tiny
DeepIce on CPU in about a minute, and loads the exported encoder back into a
stock graphnet DeepIce:
```
python examples/graphnet_demo.py --out curtain_demo_out
```
The same run is also available through the Hydra path: stage the demo inputs
once, then launch from the experiment config:
```
python examples/graphnet_demo.py --prepare-only
python examples/train_curtain.py +experiment=prometheus_demo
```

## 📥 What your setup provides
SPINE plugs into your code through four contracts. Everything else (loading
infrastructure, splits, logging, versioning) stays yours.

**1. Raw events.** Any PyTorch `Dataset` yielding
`raw[i] -> {"event_no": int, "pulses": [P, F] float32, "sensor_key": [P] int}`
(stated canonically in `spine/data/datamodule.py`). Pulses stay raw: the
sampler builds its cutoffs and dt targets in detector units, and
standardization happens later at collate. Columns follow the task's
`FeatureLayout`, by default `(x, y, z, t, charge)`; pass a different layout
instead of reordering your data. `sensor_key` is a unique integer identity
per sensor carried in the data itself; single PMT detectors can use any
stable per sensor id. Reference readers:
`spine.data.readers.SqliteRawDataset` in core, and graphnet's
`SQLiteDataset` / `LMDBDataset` adapted via
`spine_graphnet.readers.GraphNetRawDataset` (see `examples/graphnet_demo.py`).

**2. A geometry asset.** An `.npz` with per sensor arrays: `xyz [S, 3]` in
the same units as the pulse coordinates, `knn_idx [S, K]` neighbours sorted
by distance (the sampler walks it to the nearest dark sensor, so K must be
large enough that one is always found; the full sorted list is safest), and
one unique integer key array whose name you pass to
`load_geometry(path, sensor_key=...)`. The reader's `sensor_key` values must
resolve through exactly these keys; coordinate matching is deliberately not
supported. `build_geometry_asset` in `examples/graphnet_demo.py` shows a
build from a pulse file.

**3. Selections.** SPINE owns no split: `fit` takes separate train and val
Datasets and you keep them disjoint. Every selected event must be guaranteed
splittable: pre filter with `spine.pretext.curtain.sampler.can_always_split`
using the same `min_visible`/`min_future` you give the task, and float32
times. `make_sample` raises on events that slip through rather than skipping
them silently.

**4. Feature scaling.** A `FeatureScaler` subclass (`scale_pulses` and
`scale_positions`, both applying the same xyz factors) for your detector, or
wrap a graphnet detector with `spine_graphnet.scaling.DetectorScaler`. If you
fine tune in graphnet afterwards, use the standardization of the downstream
detector so the encoder sees one feature space in both stages.

Bringing your own encoder instead of DeepIce is one more contract: a
`Backbone` with `encode(batch) -> EncodedEvent` (per token embeddings, token
mask, CLS embedding) and an `out_dim` attribute; `batch["pulses"]` arrives as
a jagged NJT.
