"""Batched QPSO loop — whole swarm resident on `device`, one convergence-scalar
sync per iter (spec §5).

The math is the same as engine_numpy.qpso.run_qpso:
    alpha_t  = alpha0 - (alpha0 - alpha1) * t/T
    mbest    = pbest_theta.mean(dim=0)
    P        = phi * pbest + (1-phi) * gbest
    theta    = (P + sign * alpha * |mbest - theta| * (-log u)) mod 2π

decode + fitness happen batched across all M particles.

No diversity / no GLS / no warm-start yet — this is the parity port of the
Module-3 core (spec §8 step 6). Diversity + GLS live on CPU per §5's
guidance (branchy Or-opt / 2-opt* is a poor GPU fit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from common.bootstrap import get_logger, set_all_seeds, wallclock
from engine_torch.batch_encoding import batch_decode
from engine_torch.batch_fitness import batch_fitness
from engine_torch.device import pick_torch_device, synchronize
from engine_torch.graph_torch import TorchGraph, to_torch
from engine_numpy.graph import Graph

_log = get_logger("qpso_torch")
_TWO_PI = 2.0 * np.pi


@dataclass
class QPSOTorchResult:
    best_cost: float
    best_report_cost: float
    best_routes: List[List[int]]
    convergence: List[float] = field(default_factory=list)
    iterations: int = 0
    wall_time_s: float = 0.0
    device: str = "cpu"
    feasible: bool = False


def _extract_routes(order_row: torch.Tensor, vehicle_row: torch.Tensor,
                    K: int, depot: int = 0) -> List[List[int]]:
    """Materialize routes for one particle — CPU only, at the end of the run.

    Called once (on gbest) for the return payload; NEVER inside the QPSO loop.
    """
    order = order_row.cpu().numpy()
    vehicle = vehicle_row.cpu().numpy()
    routes: List[List[int]] = []
    for k in range(1, K + 1):
        seq = [depot]
        for pos in order:
            if vehicle[pos] == k:
                seq.append(int(pos) + 1)
        seq.append(depot)
        routes.append(seq)
    return routes


def run_qpso_torch(
    g: Graph,
    K: int,
    M: int = 20,
    T: int = 500,
    seed: int = 0,
    weights: Optional[dict] = None,
    alpha0: float = 1.0,
    alpha1: float = 0.5,
    fleet_capacity: Optional[float] = None,
    R_cap: float = 1.0e5,
    device: Optional[torch.device] = None,
    use_gpu: bool = True,
    log_every: int = 0,
) -> QPSOTorchResult:
    """Torch/GPU version of Module-3 QPSO (no diversity, no GLS)."""
    set_all_seeds(seed)
    if device is None:
        device = pick_torch_device(use_gpu=use_gpu)
    w = {"w1": 0.3, "w2": 0.5, "w3": 0.2, "w4": 0.0}
    if weights:
        w.update(weights)

    tg = to_torch(g, device=device)
    if fleet_capacity is not None:
        tg.capacity = float(fleet_capacity)

    Nc = tg.n - 1
    if Nc < 1:
        raise ValueError("need at least one customer")

    gen = torch.Generator(device="cpu")   # keep RNG on CPU for reproducibility across backends
    gen.manual_seed(int(seed))

    def _rand(shape):
        return torch.rand(shape, generator=gen, dtype=torch.float32).to(device)

    theta = (_rand((M, 2, Nc)) * _TWO_PI)   # (M, 2, Nc)
    pbest_theta = theta.clone()
    order, vehicle = batch_decode(theta[:, 0], theta[:, 1], K)
    Z_search, Z_report, feasible = batch_fitness(
        theta[:, 0], theta[:, 1], order, vehicle, tg, K, w, R_cap=R_cap,
    )
    pbest_val = Z_search.clone()
    pbest_report = Z_report.clone()

    gbest_idx = int(torch.argmin(pbest_val).item())
    gbest_theta = pbest_theta[gbest_idx].clone()          # (2, Nc)
    gbest_val = float(pbest_val[gbest_idx].item())
    gbest_report = float(pbest_report[gbest_idx].item())
    gbest_order = order[gbest_idx].clone()
    gbest_vehicle = vehicle[gbest_idx].clone()
    gbest_feasible = bool(feasible[gbest_idx].item())

    convergence: List[float] = [gbest_val]
    t0 = wallclock()

    for t in range(1, T + 1):
        alpha = alpha0 - (alpha0 - alpha1) * (t / T)
        mbest = pbest_theta.mean(dim=0)                    # (2, Nc)

        phi = _rand(theta.shape)
        u = _rand(theta.shape).clamp_min_(1e-10)
        sign_r = _rand(theta.shape)
        sign = torch.where(sign_r >= 0.5, 1.0, -1.0)

        P = phi * pbest_theta + (1.0 - phi) * gbest_theta.unsqueeze(0)
        theta = P + sign * alpha * torch.abs(mbest.unsqueeze(0) - theta) * (-torch.log(u))
        theta = torch.remainder(theta, _TWO_PI)

        order, vehicle = batch_decode(theta[:, 0], theta[:, 1], K)
        Z_search, Z_report, feasible = batch_fitness(
            theta[:, 0], theta[:, 1], order, vehicle, tg, K, w, R_cap=R_cap,
        )

        improved = Z_search < pbest_val
        if improved.any():
            pbest_val = torch.where(improved, Z_search, pbest_val)
            pbest_report = torch.where(improved, Z_report, pbest_report)
            pbest_theta = torch.where(improved.unsqueeze(-1).unsqueeze(-1), theta, pbest_theta)

            # gbest update — one scalar convergence check per iter (spec §5).
            new_best_idx = int(torch.argmin(pbest_val).item())
            new_best_val = float(pbest_val[new_best_idx].item())
            if new_best_val < gbest_val:
                gbest_val = new_best_val
                gbest_report = float(pbest_report[new_best_idx].item())
                gbest_theta = pbest_theta[new_best_idx].clone()
                gbest_order = order[new_best_idx].clone()
                gbest_vehicle = vehicle[new_best_idx].clone()
                gbest_feasible = bool(feasible[new_best_idx].item())

        convergence.append(gbest_val)
        if log_every and (t % log_every == 0 or t == T):
            synchronize(device)
            _log.info(
                "qpso-torch iter",
                extra={
                    "iteration": t, "best_cost": gbest_val,
                    "wall_time_s": round(wallclock() - t0, 4),
                    "alpha": round(alpha, 4), "feasible": gbest_feasible,
                },
            )

    synchronize(device)
    wall = wallclock() - t0
    best_routes = _extract_routes(gbest_order, gbest_vehicle, K, depot=tg.depot)
    return QPSOTorchResult(
        best_cost=gbest_val,
        best_report_cost=gbest_report,
        best_routes=best_routes,
        convergence=convergence,
        iterations=T,
        wall_time_s=wall,
        device=str(device),
        feasible=gbest_feasible,
    )
