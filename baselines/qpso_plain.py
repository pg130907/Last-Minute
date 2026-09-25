"""Plain QPSO baseline — the ablation floor.

Single-chain angle vector, no chaos, no phase rescatter, no GLS, no
rank-based decode revision. Uses `run_qpso(diversity=False, gls=False)` with
the direct-vehicle-decode path AND simulates single-chain-ness by tying
theta_seq and theta_veh together at every update (pre-revision behavior).
For the ablation we run this variant, then progressively re-enable
mechanisms; each step's mean-cost delta becomes the mechanism's honest gain
attribution.
"""
from __future__ import annotations

import time
from typing import Optional

from baselines.common import BaselineResult
from engine_numpy.graph import Graph
from engine_numpy.qpso import run_qpso


def run(
    g: Graph, K: int, M: int = 20, T: int = 300, seed: int = 0,
    weights: Optional[dict] = None, fleet_capacity: Optional[float] = None,
) -> BaselineResult:
    t0 = time.time()
    r = run_qpso(
        g, K=K, M=M, T=T, seed=seed, weights=weights,
        fleet_capacity=fleet_capacity,
        diversity=False, gls=False,
    )
    return BaselineResult(
        name="qpso_plain",
        best_cost=r.best_report_cost,
        best_cost_search=r.best_cost,
        best_routes=r.best_routes,
        wall_time_s=time.time() - t0,
        feasible=r.feasible,
        seed=seed,
        iterations=T,
        convergence=r.convergence,
    )
