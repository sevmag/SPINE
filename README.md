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
  ssl_module.py  Lightning SSLModule (optimizer/scheduler injected as factories)
  utils.py    TransferCheckpoint callback (best-val backbone export)
  train.py    reader-agnostic fit() assembly
configs/      Hydra groups: backbone/ task/ optimizer/ scheduler/ callbacks/ trainer/ data/
integrations/spine_graphnet/  graphnet integration: DeepIce backbone + reader adapter
examples/     Hydra launcher (train_curtain.py)
tests/        core-independence gate (spine imports no graphnet)
```

See `DESIGN.md` for the module decomposition, interfaces, and the decisions
behind them (graphnet surface, LMDB, selections stay the caller's, DDP gotchas).

## Runtime
Needs Python 3.10, torch 2.6 and pytorch-lightning >= 2.5 (see
`pyproject.toml`), plus a graphnet checkout on `PYTHONPATH` for the example
DeepIce backbone (graphnet is not on PyPI). Pretraining and finetuning must
share one environment: the emitted encoders are loaded back into graphnet's
DeepIce downstream.

## Running
Runs are composed with **Hydra** from `configs/` and launched via
`examples/train_curtain.py`; the core library is Hydra-free (every component is
`instantiate`d from config at the launcher). Override any group or value:
```
python examples/train_curtain.py \
    geo=geometry.npz geo_sensor_key=pmt_id out=ckpt.pth data.db=hexagon.db \
    data.train_selection=train.parquet data.val_selection=val.parquet \
    callbacks=curtain_auc task/objectives=v2 trainer.devices=4
```
## Reading data
SPINE mandates **no reader**. Provide any PyTorch `Dataset` satisfying the
contract stated canonically in `spine/data/datamodule.py`:
`raw[i] -> {"event_no": int, "pulses": [P, F] raw, "sensor_key": [P] int}` --
feature columns per the task's `FeatureLayout`, sensor keys matching the
geometry asset. A minimal SQLite reference reader ships as
`spine.data.readers.SqliteRawDataset`; GraphNeT's `LMDBDataset` /
`SQLiteDataset` are the recommended storage layers, adapted via
`spine_graphnet.readers.GraphNetRawDataset`.
