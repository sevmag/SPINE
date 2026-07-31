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
  data/       event sources (sqlite/lmdb), frozen train/val/test split, geometry, standardize
  backbones/  encoder interface + DeepIce wrapper (swappable)
  pretext/    pretext-task interface + curtain/ (the first task)
  engine/     Lightning module, optimizer/scheduler, transfer-checkpoint export
  train.py    build-from-config -> Trainer.fit
```

See `DESIGN.md` for the module decomposition, interfaces, and the decisions
behind them (graphnet surface, LMDB, the frozen-split hygiene, DDP gotchas).

## Runtime
Runs in the `graphnet_torch26_*_dirdist__unstable` env (Python 3.10, torch 2.6);
the DeepIce backbone comes from graphnet on `PYTHONPATH`. Encoders emitted here
must load in that same env on the finetuning side.
