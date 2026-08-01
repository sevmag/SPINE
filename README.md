# SPINE — Self-supervised Pretraining In Neutrino Experiments

Extensible self-supervised pretraining of transformers on neutrino-telescope
data. The repo produces **pretrained backbones** (encoder checkpoints) that
downstream supervised benchmarks load and fine-tune — a *spine* is a backbone,
which is exactly what this emits.

The first pretext task is **CURTAIN** (occupancy / light-front forecast). The
architecture is built so a new self-supervised method is a small plugin under
`pretext/`, reusing the data, backbone, and training engine unchanged.

## Layout
```
src/spine/
  data/       geometry + FeatureScaler scaling, datamodule (reader- & selection-agnostic)
  backbones/  encoder interface (swappable; graphnet-free core)
  pretext/    pretext-task interface + curtain/ (the first task)
  ssl_module.py  Lightning SSLModule + optimizer/scheduler
  utils.py    TransferCheckpoint callback (best-val backbone export)
  train.py    reader-agnostic fit() assembly
configs/      Hydra config groups: backbone/ sampler/ objectives/ trainer/ data/
examples/     graphnet integration: DeepIce backbone + reference readers + a Hydra launcher
```

See `DESIGN.md` for the module decomposition, interfaces, and the decisions
behind them (graphnet surface, LMDB, selections stay the caller's, DDP gotchas).

## Runtime
Runs in the `graphnet_torch26_*_dirdist__unstable` env (Python 3.10, torch 2.6);
the example DeepIce backbone comes from graphnet on `PYTHONPATH`. Encoders
emitted here must load in that same env on the finetuning side.

## Running
Runs are composed with **Hydra** from `configs/` and launched via
`examples/train_curtain.py`; the core library is Hydra-free (every component is
`instantiate`d from config at the launcher). Override any group or value:
```
cd examples && PYTHONPATH=../src python train_curtain.py \
    geo=geo.npz out=ckpt.pth data.db=hexagon.db \
    data.train_selection=train.parquet data.val_selection=val.parquet \
    objectives=v2 trainer.devices=4
```
## Reading data
SPINE ships **no reader**. Provide any PyTorch `Dataset` where
`dataset[i] -> {"event_no": int, "pulses": np.ndarray[P,5]}` (x,y,z,t,charge, **raw** -- SPINE standardizes after the pretext split).
Recommended: GraphNeT's `LMDBDataset` / `SQLiteDataset` with an identity
detector + `NodesAsPulses`, adapted via `examples/readers.py:GraphNetRawDataset`
(a minimal SQLite reference reader is in the same file).
