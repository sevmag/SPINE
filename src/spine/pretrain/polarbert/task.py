"""Masked DOM prediction: the self-supervised pretext of Timiryasov, Tastet &
Ruchayskiy, "PolarBERT: A Foundation Model for IceCube" (NeurIPS ML4PS 2024),
reproduced as a SPINE task after the paper's reference code
(github.com/timinar/PolarBERT).

Per event, each pulse is masked independently with probability ``mask_ratio``
(the reference's Bernoulli draw): the masked pulses' SENSOR-ID embedding is
replaced, inside ``PolarBERTBackbone``, by a learned mask vector while their
time and charge stay visible -- the inverse of masked point modeling, which
hides coordinates and shows content. A linear unembedding head classifies each
masked pulse's sensor id over the detector vocabulary (cross-entropy,
per-event mean over that event's masked pulses with the reference's epsilon
guard, then mean over events). An auxiliary head on the CLS token regresses
log10 of the event's total charge (MSE), a quantity not linearly extractable
from the log-charges of individual pulses; the total loss is
``CE + lambda_charge * MSE`` with the paper's default lambda 1.

The reference masks only "primary" (non-auxiliary) pulses; NuBench hexagon
data has no auxiliary flag, so every pulse is eligible. Feature conventions
follow this project (charge-weighted-mean-referenced time in microseconds,
log10(1+q) charge) rather than the reference's Kaggle-window constants; the
charge-regression target is log10 of the summed RAW charge of the FULL event,
computed before the pulse cap so a subsampled event keeps the same target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spine.pretrain.base import PretrainTask, Sample

_TIME_REFS = ("min", "cwm", None)
_EPS = 1e-8  # the reference's guard for events whose Bernoulli draw masked nothing


class PolarBERTHead(nn.Module):
    """Sensor-id unembedding over pulse tokens + total-charge head on CLS."""

    def __init__(self, dim: int, num_sensors: int):
        """Build the classification and regression heads.

        Args:
            dim: Encoder token width.
            num_sensors: Sensor vocabulary size; logits span num_sensors + 1
                (index 0 is the padding class, never a target).
        """
        super().__init__()
        self.unembedding = nn.Linear(dim, num_sensors + 1)
        self.charge = nn.Linear(dim, 1)

    def forward(self, _query_pos: Tensor, enc: Any) -> tuple[Tensor, Tensor]:
        """Classify every pulse token's sensor id; regress charge from CLS.

        The query-position argument the engine passes is unused: the pretext
        predicts one class per encoder token plus one scalar per event.
        """
        return self.unembedding(enc.tokens), self.charge(enc.cls)


class PolarBERTTask(PretrainTask):
    """Masked DOM prediction + total-charge regression (paper-faithful)."""

    objectives: list = []

    def __init__(
        self,
        geo: dict,
        scaler: Any,
        max_pulses: int = 768,
        time_ref: str | None = "cwm",
        time_scale: float = 1000.0,
        mask_ratio: float = 0.25,
        lambda_charge: float = 1.0,
        num_sensors: int = 5160,
    ):
        """Assemble the masked-DOM-prediction task.

        Args:
            geo: Geometry asset (unused; sensor ids come from the pulses).
            scaler: Detector feature scaling (unused; this task builds its
                two-feature tokens directly so the time reference can be
                taken on raw times).
            max_pulses: Cap on pulses fed to the encoder per event (random
                subsample; the paper downsamples to a fixed budget too).
            time_ref: Per-event time origin subtracted before scaling:
                "cwm" (charge-weighted mean, this project's convention),
                "min" (earliest hit) or None (raw times).
            time_scale: Divisor (ns per unit) mapping referenced time to the
                encoder input unit; 1000 = microseconds.
            mask_ratio: Per-pulse Bernoulli masking probability (paper: 0.25).
            lambda_charge: Weight of the charge-regression MSE term.
            num_sensors: Sensor vocabulary size (hexagon: 5160).

        Raises:
            ValueError: If time_ref is not a recognized option.
        """
        if time_ref not in _TIME_REFS:
            raise ValueError(f"time_ref must be one of {_TIME_REFS}, got {time_ref!r}")
        self.geo = geo
        self.scaler = scaler
        self.max_pulses = max_pulses
        self.time_ref = time_ref
        self.time_scale = time_scale
        self.mask_ratio = mask_ratio
        self.lambda_charge = lambda_charge
        self.num_sensors = num_sensors

    def make_sample(self, event: dict[str, np.ndarray], rng: np.random.Generator) -> Sample:
        """Reference times, cap pulses jointly with ids, draw the Bernoulli mask."""
        p = np.asarray(event["pulses"], dtype=np.float32)
        ids = np.asarray(event["sensor_key"], dtype=np.int64)
        if len(p) == 0:
            raise ValueError(f"event {event['event_no']} has no pulses")
        lay = self.scaler.layout
        t_raw = p[:, lay.t]
        q_raw = np.clip(p[:, lay.charge], 0.0, None)
        # Reference and charge target are taken over the whole event, before
        # the cap, so a subsampled event keeps the same origin and target.
        if self.time_ref == "min":
            t0 = float(t_raw.min())
        elif self.time_ref == "cwm":
            w = q_raw + 1e-6
            t0 = float((w * t_raw).sum() / w.sum())
        else:
            t0 = 0.0
        charge_target = float(np.log10(q_raw.sum() + 1e-6))
        if len(p) > self.max_pulses:
            keep = rng.choice(len(p), self.max_pulses, replace=False)
            keep.sort()
            t_raw, q_raw, ids = t_raw[keep], q_raw[keep], ids[keep]
        feats = np.stack(
            [(t_raw - t0) / self.time_scale, np.log10(1.0 + q_raw)], axis=1
        ).astype(np.float32)
        dom_mask = rng.random(len(feats)) < self.mask_ratio
        return dict(
            features=feats,
            dom_ids=ids,
            dom_mask=dom_mask,
            charge_target=np.float32(charge_target),
        )

    def collate(self, samples: list[Sample]) -> dict:
        """Pack features/ids jagged; mask and id targets padded."""

        def jag(tensors):
            return torch.nested.nested_tensor(tensors, layout=torch.jagged)

        def pad(arrays, fill):
            out = torch.full((len(arrays), lmax), fill, dtype=arrays[0].dtype)
            for b, a in enumerate(arrays):
                out[b, : len(a)] = a
            return out

        feats = [torch.from_numpy(s["features"]) for s in samples]
        lmax = max(t.shape[0] for t in feats)
        # qpos/label satisfy the engine's collate contract: qpos is unused by
        # PolarBERTHead; label sizes the logged batch as the number of events.
        return dict(
            features=jag(feats),
            dom_ids=jag([torch.from_numpy(s["dom_ids"]) for s in samples]),
            qpos=jag([t.clone() for t in feats]),
            label=jag([torch.ones(1) for _ in samples]),
            dom_mask=pad([torch.from_numpy(s["dom_mask"]) for s in samples], False),
            dom_target=pad([torch.from_numpy(s["dom_ids"]) for s in samples], 0),
            charge_target=torch.tensor([s["charge_target"] for s in samples]),
        )

    def build_head(self, dim: int) -> nn.Module:
        """Construct the unembedding + charge heads."""
        return PolarBERTHead(dim, self.num_sensors)

    def loss(self, output: Any, batch: dict) -> tuple[Tensor, dict[str, float]]:
        """CE over masked pulses' sensor ids + lambda * MSE on log total charge.

        The CE reduction mirrors the reference: per-event mean over masked
        pulses with an epsilon guard, then a mean over ALL events (an event
        whose draw masked nothing contributes ~zero rather than being
        dropped).
        """
        logits, charge_hat = output
        length = logits.shape[1]
        valid = batch["dom_mask"].new_zeros(logits.shape[0], length, dtype=torch.bool)
        lengths = batch["features"].offsets().diff()
        valid = torch.arange(length, device=logits.device)[None] < lengths[:, None]
        m = batch["dom_mask"][:, :length].to(logits.device) & valid
        # ids shift +1 to match the embedding table (0 = padding class)
        target = batch["dom_target"][:, :length].to(logits.device) + 1
        ce = F.cross_entropy(logits.permute(0, 2, 1), target, reduction="none")
        ce = ((ce * m).sum(1) / (m.sum(1) + _EPS)).mean()
        mse = F.mse_loss(charge_hat.squeeze(-1), batch["charge_target"].to(charge_hat.device))
        total = ce + self.lambda_charge * mse
        return total, {
            "loss_pbert_dom": float(ce.detach()),
            "loss_pbert_charge": float(mse.detach()),
        }
