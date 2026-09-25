"""Module 2 gate — fitness ALWAYS finite, even when a route crosses a non-edge.

Rationale (spec §2.5): one inf-fitness particle destroys mbest = mean(pbest).
The guard must convert inf lookups into BIG_FINITE and the aggregate must
still be a plain float — never inf, never NaN.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from engine_numpy.fitness import evaluate  # noqa: E402
from engine_numpy.graph import BIG_FINITE, build_from_dist  # noqa: E402


def _sparse_graph() -> "any":
    # 5 nodes; deliberately remove edges 1↔3 and 2↔4 so any route that hops
    # across them crosses a non-edge.
    inf = np.inf
    d = np.array([
        [0,   3, 4, inf, 6],
        [3,   0, 2, inf, 5],
        [4,   2, 0, 3,   inf],
        [inf, inf, 3, 0, 2],
        [6,   5, inf, 2, 0],
    ], dtype=np.float64)
    demand = np.array([0, 1, 1, 1, 1], dtype=np.float32)
    return build_from_dist(d, demand=demand, capacity=100.0)


def test_fitness_finite_when_route_crosses_non_edge() -> None:
    g = _sparse_graph()
    # Route 0→1→3 hops across the non-edge 1→3 (dist=inf).
    routes = [[0, 1, 3, 0]]
    r = evaluate(g, routes, weights={"w1": 1.0, "w2": 0.0, "w3": 0.0})
    # Total distance must be finite AND at least BIG_FINITE (the guard fired).
    assert math.isfinite(r.Z_search), f"Z_search not finite: {r.Z_search}"
    assert math.isfinite(r.Z_report), f"Z_report not finite: {r.Z_report}"
    assert math.isfinite(r.total_distance)
    assert r.total_distance >= BIG_FINITE, (
        f"finite guard didn't fire — got dist={r.total_distance}"
    )


def test_mbest_style_mean_finite_across_particles() -> None:
    """Simulate the mbest computation the QPSO loop will do."""
    g = _sparse_graph()
    routes_good = [[0, 1, 2, 3, 4, 0]]     # all real edges
    routes_bad = [[0, 1, 3, 4, 2, 0]]      # crosses non-edges twice
    pbest_vals = np.array([
        evaluate(g, routes_good, weights={"w1": 1.0, "w2": 0.0, "w3": 0.0}).Z_search,
        evaluate(g, routes_bad,  weights={"w1": 1.0, "w2": 0.0, "w3": 0.0}).Z_search,
    ])
    assert np.isfinite(pbest_vals).all(), pbest_vals
    mbest = pbest_vals.mean()
    assert math.isfinite(mbest), f"mbest not finite: {mbest}"


def test_finite_guard_on_time_and_emissions() -> None:
    g = _sparse_graph()
    routes = [[0, 2, 4, 0]]                # 2→4 is a non-edge
    r = evaluate(g, routes, weights={"w1": 0.0, "w2": 0.5, "w3": 0.5})
    assert math.isfinite(r.total_time)
    assert math.isfinite(r.total_emissions)
    assert r.total_time >= BIG_FINITE or r.total_emissions >= BIG_FINITE, (
        "expected at least one of time/emissions to hit BIG_FINITE"
    )


if __name__ == "__main__":
    test_fitness_finite_when_route_crosses_non_edge()
    test_mbest_style_mean_finite_across_particles()
    test_finite_guard_on_time_and_emissions()
    print("test_finite_fitness: OK")
