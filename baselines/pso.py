"""Plain PSO (Kennedy–Eberhart) on the same dual-chain encoding.

Uses velocity + position, no quantum-inspired probability well. Same decode
+ fitness as MC-GQPSO so the comparison is apples-to-apples on the
representation, not the algorithm.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from baselines.common import BaselineResult
from common.bootstrap import set_all_seeds
from engine_numpy.encoding import decode, random_init, wrap_angles
from engine_numpy.fitness import evaluate
from engine_numpy.graph import Graph

_TWO_PI = 2.0 * np.pi


def run(
    g: Graph, K: int, M: int = 20, T: int = 300, seed: int = 0,
    weights: Optional[dict] = None, fleet_capacity: Optional[float] = None,
    w: float = 0.72, c1: float = 1.49, c2: float = 1.49,
) -> BaselineResult:
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    Nc = g.n - 1
    t0 = time.time()

    theta = np.zeros((M, 2, Nc), dtype=np.float32)
    for i in range(M):
        s, v = random_init(Nc, rng)
        theta[i, 0] = s; theta[i, 1] = v
    vel = rng.uniform(-0.5, 0.5, size=theta.shape).astype(np.float32)

    def _eval(t_i):
        routes, _, _ = decode(t_i[0], t_i[1], K, depot=g.depot)
        return evaluate(g, routes, weights=weights, fleet_capacity=fleet_capacity), routes

    pbest = theta.copy()
    pbest_val = np.full(M, np.inf)
    pbest_report = np.full(M, np.inf)
    pbest_routes: list = [None] * M
    for i in range(M):
        res, routes = _eval(theta[i])
        pbest_val[i] = res.Z_search; pbest_report[i] = res.Z_report; pbest_routes[i] = routes

    g_idx = int(np.argmin(pbest_val))
    gbest = pbest[g_idx].copy()
    gbest_val = float(pbest_val[g_idx]); gbest_report = float(pbest_report[g_idx])
    gbest_routes = pbest_routes[g_idx]
    conv = [gbest_val]

    for _ in range(T):
        r1 = rng.uniform(0, 1, size=theta.shape).astype(np.float32)
        r2 = rng.uniform(0, 1, size=theta.shape).astype(np.float32)
        vel = w * vel + c1 * r1 * (pbest - theta) + c2 * r2 * (gbest[None] - theta)
        theta = wrap_angles(theta + vel).astype(np.float32)
        for i in range(M):
            res, routes = _eval(theta[i])
            if res.Z_search < pbest_val[i]:
                pbest_val[i] = res.Z_search; pbest_report[i] = res.Z_report
                pbest[i] = theta[i]; pbest_routes[i] = routes
                if res.Z_search < gbest_val:
                    gbest_val = float(res.Z_search); gbest_report = float(res.Z_report)
                    gbest = theta[i].copy(); gbest_routes = routes
        conv.append(gbest_val)

    return BaselineResult(
        name="pso", best_cost=gbest_report, best_cost_search=gbest_val,
        best_routes=gbest_routes or [], wall_time_s=time.time() - t0,
        feasible=(gbest_val == gbest_report), seed=seed, iterations=T, convergence=conv,
    )
