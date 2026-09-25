"""Simulated annealing on the same permutation + K-way split representation.

Neighbourhood: swap two customers (position exchange) OR jitter one split
point. Standard Metropolis acceptance with exponential cooling.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from baselines.common import BaselineResult
from baselines.ga import _chromo_to_routes
from common.bootstrap import set_all_seeds
from engine_numpy.fitness import evaluate
from engine_numpy.graph import Graph


def run(
    g: Graph, K: int, T_iters: int = 3000, seed: int = 0,
    weights: Optional[dict] = None, fleet_capacity: Optional[float] = None,
    T_start: float = 1000.0, T_end: float = 1.0,
    **_kwargs,
) -> BaselineResult:
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    Nc = g.n - 1
    t0 = time.time()

    perm = rng.permutation(np.arange(1, Nc + 1))
    cuts = np.sort(rng.integers(0, Nc + 1, size=K - 1)) if K > 1 else np.array([Nc], dtype=int)
    cuts = cuts.astype(int)

    def _eval(p, c):
        routes = _chromo_to_routes(p, c, K, g.depot)
        return evaluate(g, routes, weights=weights, fleet_capacity=fleet_capacity), routes

    cur_res, cur_routes = _eval(perm, cuts)
    cur_cost = cur_res.Z_search
    best_cost = cur_cost
    best_report = cur_res.Z_report
    best_routes = cur_routes
    conv = [best_cost]

    for step in range(T_iters):
        T = T_start * (T_end / T_start) ** (step / max(1, T_iters - 1))
        # Neighbour.
        if rng.uniform() < 0.7 or K == 1:
            x, y = rng.integers(0, Nc, size=2)
            new_perm = perm.copy()
            new_perm[x], new_perm[y] = new_perm[y], new_perm[x]
            new_cuts = cuts
        else:
            new_perm = perm
            new_cuts = cuts.copy()
            which = int(rng.integers(0, K - 1))
            new_cuts[which] = int(np.clip(new_cuts[which] + rng.integers(-2, 3), 0, Nc))

        res, routes = _eval(new_perm, new_cuts)
        d = res.Z_search - cur_cost
        if d < 0 or rng.uniform() < math.exp(-d / max(T, 1e-9)):
            perm, cuts = new_perm, new_cuts
            cur_cost = res.Z_search
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_report = res.Z_report
                best_routes = routes
        conv.append(best_cost)

    return BaselineResult(
        name="sa", best_cost=best_report, best_cost_search=best_cost,
        best_routes=best_routes or [], wall_time_s=time.time() - t0,
        feasible=(best_cost == best_report), seed=seed, iterations=T_iters, convergence=conv,
    )
