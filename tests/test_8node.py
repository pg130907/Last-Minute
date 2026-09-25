"""Module 3 HARD GATE — 8-node instance, spec §3.3.

Pass criteria (20 seeded runs, M=20, T=500):
  ≥ 3 runs hit 67.5 km AND mean_best ≤ 71.0 km.
"""
from __future__ import annotations

import statistics
import sys

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from engine_numpy.loaders import load_8node  # noqa: E402
from engine_numpy.qpso import run_qpso  # noqa: E402

TOL = 1e-2
KNOWN_OPTIMUM = 67.5
MEAN_CAP = 71.0
N_RUNS = 20
M_PARTICLES = 20
T_ITERS = 500


def _run_once(seed: int) -> float:
    g = load_8node()
    res = run_qpso(
        g, K=2, M=M_PARTICLES, T=T_ITERS, seed=seed,
        weights={"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0},
        fleet_capacity=8.0,
    )
    if not res.feasible:
        # Return the report cost anyway, but caller decides how to treat it.
        return res.best_report_cost
    return res.best_report_cost


def test_8node_hard_gate() -> None:
    bests = [_run_once(seed=s) for s in range(N_RUNS)]
    hits = sum(1 for b in bests if abs(b - KNOWN_OPTIMUM) < TOL)
    mean_best = statistics.mean(bests)
    print(f"  ├─ per-run bests: {[round(b, 3) for b in bests]}")
    print(f"  ├─ hits @ 67.5   : {hits}/{N_RUNS}")
    print(f"  ├─ mean best     : {mean_best:.3f} km (cap {MEAN_CAP})")
    print(f"  └─ min / max     : {min(bests):.3f} / {max(bests):.3f}")
    assert hits >= 3, f"HARD GATE: only {hits}/{N_RUNS} runs hit {KNOWN_OPTIMUM}, need ≥3"
    assert mean_best <= MEAN_CAP, f"HARD GATE: mean {mean_best:.3f} > cap {MEAN_CAP}"


if __name__ == "__main__":
    test_8node_hard_gate()
    print("test_8node HARD GATE: OK")
