"""OR-Tools CP-SAT-backed VRP baseline (spec §5c primary reference).

This is what MC-GQPSO is expected to be *close to* on static Solomon-style
quality — not to beat. Defensible wins for us are warm-start adaptation
speed, anytime behaviour, and consistency across seeds (spec §5c honesty).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from baselines.common import BaselineResult
from engine_numpy.fitness import evaluate
from engine_numpy.graph import BIG_FINITE, Graph


def run(
    g: Graph, K: int, seed: int = 0,
    weights: Optional[dict] = None, fleet_capacity: Optional[float] = None,
    time_limit_s: int = 5,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
    metaheuristic: str = "GUIDED_LOCAL_SEARCH",
    **_kwargs,
) -> BaselineResult:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    t0 = time.time()

    N = g.n
    # OR-Tools wants integer costs; scale km × 1000 → metres.
    dist_int = np.where(g.adj_mask | np.eye(N, dtype=bool), g.dist, BIG_FINITE)
    dist_int = np.rint(dist_int * 1000.0).astype(np.int64)
    demand = (g.demand if g.demand is not None else np.zeros(N)).astype(int)
    cap = int(fleet_capacity if fleet_capacity is not None else (g.capacity or 0))

    mgr = pywrapcp.RoutingIndexManager(N, K, g.depot)
    routing = pywrapcp.RoutingModel(mgr)

    def _dist_cb(fi, ti):
        return int(dist_int[mgr.IndexToNode(fi), mgr.IndexToNode(ti)])
    dist_idx = routing.RegisterTransitCallback(_dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    if cap > 0:
        def _demand_cb(fi):
            return int(demand[mgr.IndexToNode(fi)])
        dem_idx = routing.RegisterUnaryTransitCallback(_demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            dem_idx, 0, [cap] * K, True, "Capacity",
        )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution_strategy,
    )
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic,
    )
    params.time_limit.seconds = int(time_limit_s)
    params.log_search = False

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return BaselineResult(
            name="ortools", best_cost=float("inf"), best_cost_search=float("inf"),
            best_routes=[], wall_time_s=time.time() - t0, feasible=False,
            seed=seed, notes="ortools returned no solution",
        )

    routes: list[list[int]] = []
    for k in range(K):
        idx = routing.Start(k)
        r = [mgr.IndexToNode(idx)]
        while not routing.IsEnd(idx):
            idx = sol.Value(routing.NextVar(idx))
            r.append(mgr.IndexToNode(idx))
        routes.append(r)

    # Score via our own fitness so the comparison is on the SAME objective.
    fit = evaluate(g, routes, weights=weights, fleet_capacity=fleet_capacity)
    return BaselineResult(
        name="ortools", best_cost=fit.Z_report, best_cost_search=fit.Z_search,
        best_routes=routes, wall_time_s=time.time() - t0, feasible=fit.feasible,
        seed=seed, iterations=0, notes=f"{metaheuristic}/{first_solution_strategy}",
    )
