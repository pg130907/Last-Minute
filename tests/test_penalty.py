"""Module 2 gate — overload strictly worse than feasible; late strictly worse.

Also: Z_report ignores penalties; Z_search adds them.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from engine_numpy.fitness import DEFAULT_R_CAP, DEFAULT_R_TW, evaluate  # noqa: E402
from engine_numpy.graph import build_from_dist  # noqa: E402
from engine_numpy.loaders import load_8node  # noqa: E402


def _known_optimum_routes() -> list[list[int]]:
    # Spec §3.3 optimum on the 8-node instance.
    return [[0, 4, 7, 6, 0], [0, 1, 3, 5, 8, 2, 0]]


def test_overload_strictly_worse() -> None:
    g = load_8node()
    routes = _known_optimum_routes()
    feasible = evaluate(g, routes, fleet_capacity=8.0)
    # Second route loads = 1+1+1+2+2 = 7 ≤ 8 → feasible.
    assert feasible.feasible, f"expected feasible baseline; got violations={feasible.violations}"

    # Same routes but with a tighter fleet cap of 4 → both routes overload.
    overloaded = evaluate(g, routes, fleet_capacity=4.0)
    assert not overloaded.feasible
    assert overloaded.Z_search > feasible.Z_search, (
        f"overload not penalized: feas={feasible.Z_search}, ovl={overloaded.Z_search}"
    )
    # Report cost is identical (no penalties in Z_report).
    assert abs(overloaded.Z_report - feasible.Z_report) < 1e-3
    # Penalty magnitude sanity.
    total_over = sum(v["amount"] for v in overloaded.violations.overload)
    assert abs(overloaded.Z_search - feasible.Z_search - DEFAULT_R_CAP * total_over) < 1e-3


def test_late_strictly_worse_and_early_is_free() -> None:
    # Build a tiny 4-node grid with tight windows on customer 2.
    d = np.array([
        [0, 5, 5, 5],
        [5, 0, 5, 5],
        [5, 5, 0, 5],
        [5, 5, 5, 0],
    ], dtype=np.float64)
    demand = np.array([0, 1, 1, 1], dtype=np.float32)
    # tau0 = dist/50 * 60 = 6 min per hop. Route [0,1,2,3,0] arrives at 2 around min 12.
    tw_early = np.array([0.0, 0.0, 20.0, 0.0], dtype=np.float32)   # early @ node 2
    tw_late = np.array([1e6, 1e6, 30.0, 1e6], dtype=np.float32)    # comfortable
    service = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    g = build_from_dist(
        d, demand=demand, tw_early=tw_early, tw_late=tw_late,
        service_time=service, capacity=100.0,
    )
    routes = [[0, 1, 2, 3, 0]]

    feasible = evaluate(g, routes)
    assert feasible.feasible, f"baseline should be feasible; viol={feasible.violations}"

    # Early arrival must NOT be penalized. Node 2's arrival ≈ 6+1+6 = 13, e=20 → waited.
    r = feasible.per_route[0]
    idx_of_2 = r.sequence.index(2)
    assert r.arrival_times[idx_of_2] < tw_early[2], "arrival must be BEFORE e_2 (proving we waited)"
    assert r.depart_times[idx_of_2] >= tw_early[2], "depart must respect wait"

    # Now tighten l_2 so we're 5 min late; verify Z_search jumps by ~R_tw*5.
    tw_late_tight = tw_late.copy()
    tw_late_tight[2] = 8.0     # arrival ≈ 13 → 5 min late
    g2 = build_from_dist(
        d, demand=demand, tw_early=tw_early, tw_late=tw_late_tight,
        service_time=service, capacity=100.0,
    )
    late = evaluate(g2, routes)
    assert not late.feasible
    assert late.Z_search > feasible.Z_search, "late arrival not penalized"
    total_late = sum(v["minutes_late"] for v in late.violations.late)
    assert total_late >= 4.9
    delta = late.Z_search - late.Z_report - (feasible.Z_search - feasible.Z_report)
    assert abs(delta - DEFAULT_R_TW * total_late) < 1e-2


def test_z_report_matches_known_optimum_distance() -> None:
    """On 8-node, the optimum routes should give total distance == 67.5 km."""
    g = load_8node()
    r = evaluate(g, _known_optimum_routes(), weights={"w1": 1.0, "w2": 0.0, "w3": 0.0})
    assert abs(r.total_distance - 67.5) < 1e-3, f"expected 67.5 km, got {r.total_distance}"
    assert abs(r.Z_report - 67.5) < 1e-3


if __name__ == "__main__":
    test_overload_strictly_worse()
    test_late_strictly_worse_and_early_is_free()
    test_z_report_matches_known_optimum_distance()
    print("test_penalty: OK")
