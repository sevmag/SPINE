"""Hydra launcher for CURTAIN pretraining -- copy and adapt.

Composes configs/ and instantiates every component; the core is Hydra-free.
Needs spine + spine_graphnet installed (`pip install -e .` in the graphnet
env). You bring disjoint train/val selection parquets (SPINE owns no split)
and a geometry asset with a sensor-key array. Override anything:

  python examples/train_curtain.py geo=... out=... data.db=... \
      data.train_selection=... data.val_selection=... geo_sensor_key=pmt_id \
      task/objectives=v2 trainer.devices=4
"""

from __future__ import annotations

import hydra
import pyarrow.parquet as pq
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from spine import train as spine_train
from spine.data.geometry import load_geometry


def load_event_nos(path: str, column: str = "event_no"):
    """Read the event ids of a selection parquet you built.

    Args:
        path: The selection parquet.
        column: Column holding the event ids.

    Returns:
        The event ids, in file order.
    """
    return pq.read_table(path, columns=[column])[column].to_numpy()


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Build every component from the composed config and run fit().

    Args:
        cfg: The Hydra-composed run configuration.
    """
    geo = load_geometry(cfg.geo, sensor_key=cfg.geo_sensor_key)

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
    wandb_cfg = OmegaConf.to_container(t.wandb, resolve=True) if t.wandb else None
    sched = cfg.get("scheduler")
    spine_train.fit(
        train_raw,
        val_raw,
        task,
        backbone,
        cfg.out,
        optimizer=instantiate(cfg.optimizer),
        scheduler=instantiate(sched) if sched else None,
        scheduler_config=(
            OmegaConf.to_container(cfg.scheduler_config, resolve=True)
            if cfg.scheduler_config
            else None
        ),
        batch=t.batch,
        num_workers=t.num_workers,
        val_num_workers=t.val_num_workers,
        devices=t.devices,
        precision=t.precision,
        max_epochs=t.max_epochs,
        patience=t.patience,
        grad_clip=t.grad_clip,
        callbacks=[instantiate(c) for c in (cfg.get("callbacks") or [])],
        wandb=wandb_cfg,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


if __name__ == "__main__":
    main()
