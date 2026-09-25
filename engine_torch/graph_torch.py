"""Numpy Graph → resident torch tensors, finite-guarded (spec §2.5).

Non-edges (dist/tau0=inf) are substituted with BIG_FINITE up front so the
batched fitness never has to branch on isinf inside the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from engine_numpy.graph import BIG_FINITE, Graph

_F32 = torch.float32
_I64 = torch.int64


@dataclass
class TorchGraph:
    """All tensors resident on `device`. float32 per spec §5."""
    dist: torch.Tensor           # (N, N)
    tau: torch.Tensor            # (N, N) — current BPR slice
    emis: torch.Tensor           # (N, N)
    demand: torch.Tensor         # (N,)  — depot demand is 0
    capacity: float              # per-vehicle Q (uniform fleet)
    depot: int
    tw_early: Optional[torch.Tensor] = None    # (N,)
    tw_late: Optional[torch.Tensor] = None
    service_time: Optional[torch.Tensor] = None
    device: torch.device = torch.device("cpu")

    @property
    def n(self) -> int:
        return int(self.dist.shape[0])


def to_torch(g: Graph, device: torch.device) -> TorchGraph:
    """Copy a numpy Graph onto `device`, finite-guarding every matrix."""
    def _guard(a: np.ndarray) -> torch.Tensor:
        arr = np.where(np.isfinite(a), a, BIG_FINITE).astype(np.float32)
        return torch.from_numpy(arr).to(device)

    N = g.n
    demand = g.demand if g.demand is not None else np.zeros(N, dtype=np.float32)
    dist = _guard(g.dist)
    tau = _guard(g.tau)
    emis = _guard(g.emis)     # already BIG_FINITE-guarded on non-edges

    return TorchGraph(
        dist=dist,
        tau=tau,
        emis=emis,
        demand=torch.from_numpy(np.asarray(demand, dtype=np.float32)).to(device),
        capacity=float(g.capacity) if g.capacity is not None else 0.0,
        depot=int(g.depot),
        tw_early=None if g.tw_early is None else torch.from_numpy(
            np.asarray(g.tw_early, dtype=np.float32)).to(device),
        tw_late=None if g.tw_late is None else torch.from_numpy(
            np.asarray(g.tw_late, dtype=np.float32)).to(device),
        service_time=None if g.service_time is None else torch.from_numpy(
            np.asarray(g.service_time, dtype=np.float32)).to(device),
        device=device,
    )
