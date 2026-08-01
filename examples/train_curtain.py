"""Runnable CURTAIN pretraining example -- Hydra-composed.

Configs live in the top-level configs/ (composed here). SPINE's core is
Hydra-free; this launcher instantiates every component from config via
`hydra.utils.instantiate` and calls spine.train.fit. The graphnet-specific
targets (DeepIceBackbone, SqliteRawDataset) resolve because this file's dir
(examples/) is on sys.path; the core targets need src/ on PYTHONPATH.

You bring disjoint train/val selection parquets (SPINE owns no split). Override
anything on the CLI, e.g.:

  PYTHONPATH=../src python train_curtain.py \
      geo=/.../hexagon_geometry.npz out=/.../ckpt.pth \
      data.db=/.../hexagon.db \
      data.train_selection=/.../train.parquet \
      data.val_selection=/.../val.parquet \
      task/objectives=v2 trainer.devices=4 trainer.precision=bf16-mixed
"""

from __future__ import annotations

import hydra
import pyarrow.parquet as pq
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from spine import train as spine_train
from spine.data.geometry import load_geometry


def load_event_nos(path: str, column: str = "event_no"):
    """event_no array from a selection parquet you built."""
    return pq.read_table(path, columns=[column])[column].to_numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    geo = load_geometry(cfg.geo)

    d = cfg.data
    train_ev = load_event_nos(d.train_selection)
    if d.n_train:
        train_ev = train_ev[: d.n_train]
    val_ev = load_event_nos(d.val_selection)
    train_raw = instantiate(d.reader, event_nos=train_ev)
    val_raw = instantiate(d.reader, event_nos=val_ev)

    # cfg.task is the whole CURTAIN plugin config -- recursive instantiate builds
    # its sampler + objectives too; only the runtime objects are injected here
    task = instantiate(cfg.task, geo=geo, scaler=instantiate(cfg.scaler))
    backbone = instantiate(cfg.backbone)

    t = cfg.trainer
    spine_train.fit(
        train_raw, val_raw, task, backbone, cfg.out,
        batch=t.batch, lr=t.lr, num_workers=t.num_workers, devices=t.devices,
        precision=t.precision, max_epochs=t.max_epochs, patience=t.patience,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


if __name__ == "__main__":
    main()
