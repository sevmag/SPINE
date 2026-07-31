"""Raw pulse/sensor features -> standardized encoder features (NuBench Hexagon).

Explicit and separate so it runs at the model boundary, AFTER the pretext split
-- the pretext must see *raw* values (CURTAIN's Delta-t reference is the
charge-weighted mean time of raw pulses; the geometry match is in raw metres).
Matches graphnet's Hexagon detector scaling. If the backbone instead reads
through a graphnet GraphDefinition with a real detector, prefer that detector
over this hand copy (removes the must-match-by-hand coupling).
"""

import torch
from torch import Tensor

_POS = torch.tensor([100.0, 100.0, 1000.0])


def standardize_pulses(x: Tensor) -> Tensor:
    """Raw [., 5] (x, y, z, t, charge) -> standardized encoder features."""
    out = torch.empty_like(x)
    out[..., :3] = x[..., :3] / _POS.to(x.device)
    out[..., 3] = x[..., 3] / 1e6
    out[..., 4] = torch.log10(1.0 + x[..., 4].clamp(min=1e-2))
    return out


def standardize_pos(p: Tensor) -> Tensor:
    """Raw [., 3] sensor positions -> standardized coordinates."""
    return p / _POS.to(p.device)
