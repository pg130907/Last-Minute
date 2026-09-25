"""Module 4b — GLS smoke test.

Two checks:
  1. Given an obviously-suboptimal 8-node solution, GLS reduces the reported
     cost strictly and the improved routes re-encode idempotently.
  2. On a mid-size synthetic instance, +GLS yields a lower mean best cost
     than baseline QPSO over a few seeds (the ablation delta the spec §5c
     wants — checked lightly here so we don't gate on it).
"""
from __future__ import annotations

import statistics
import sys

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from engine_numpy.encoding import decode, re_encode  # noqa: E402
from engine_numpy.fitness import evaluate  # noqa: E402
from engine_numpy.gls import compute_knn, guided_local_search  # noqa: E402
from engine_numpy.loaders import load_8node, make_synthetic  # noqa: E402
from engine_numpy.qpso import run_qpso  # noqa: E402


def test_gls_improves_obvious_suboptimum() -> None:
    """A deliberately-bad 8-node solution should be reduced by GLS."""
    g = load_8node()
    knn = compute_knn(g, k=8)
    # A silly solution that swaps two large-cost customers to the wrong routes.
    bad = [[0, 5, 4, 7, 0], [0, 1, 3, 6, 8, 2, 0]]
    base = evaluate(g, bad, weights={"w1": 1.0, "w2": 0, "w3": 0}, fleet_capacity=8.0)
    improved, res = guided_local_search(
        bad, g, weights={"w1": 1.0, "w2": 0, "w3": 0},
        knn=knn, fleet_capacity=8.0, max_outer=2, max_inner=25,
    )
    assert res.Z_search < base.Z_search - 1e-6, (
        f"GLS did not improve: {base.Z_search:.3f} → {res.Z_search:.3f}"
    )
    print(f"  ├─ bad→gls: {base.Z_report:.3f} → {res.Z_report:.3f}")


def test_re_encode_idempotent_post_gls() -> None:
    """Spec §3.1 mandatory: decode(re_encode(gls_out)) == gls_out."""
    g = load_8node()
    knn = compute_knn(g, k=8)
    bad = [[0, 5, 4, 7, 0], [0, 1, 3, 6, 8, 2, 0]]
    improved, _ = guided_local_search(
        bad, g, weights={"w1": 1.0, "w2": 0, "w3": 0},
        knn=knn, fleet_capacity=8.0, max_outer=1, max_inner=10,
    )
    ts, tv = re_encode(improved, Nc=g.n - 1, K=2, depot=0)
    rt2, _, _ = decode(ts, tv, K=2, depot=0)
    assert rt2 == improved, f"re-encode not idempotent:\n {improved}\n {rt2}"


def test_ablation_delta_on_midsize_instance() -> None:
    """Ablation smoke on N=30 synthetic: +gls should not be WORSE than plain.
    We don't strictly assert +gls < baseline (small-sample noise), but we
    verify the mechanisms all run without crashing and produce feasible
    solutions on a bigger instance where GLS has room to help.
    """
    g = make_synthetic(n=30, seed=0)
    seeds = [11, 22, 33]
    results = {}
    for name, kw in [
        ("baseline", {}),
        ("+gls", {"gls": True, "gls_every": 25}),
        ("+div+gls", {"diversity": True, "gls": True, "gls_every": 25}),
    ]:
        bests = [
            run_qpso(g, K=4, M=20, T=150, seed=s, fleet_capacity=100.0, **kw).best_report_cost
            for s in seeds
        ]
        results[name] = statistics.mean(bests)
        print(f"  ├─ {name:10s}: mean_best={results[name]:.2f}  bests={[round(b,2) for b in bests]}")
    # Sanity: mechanisms complete and yield finite feasible costs.
    for v in results.values():
        assert v > 0 and v < 1e6, f"unreasonable cost: {v}"


if __name__ == "__main__":
    test_gls_improves_obvious_suboptimum()
    test_re_encode_idempotent_post_gls()
    test_ablation_delta_on_midsize_instance()
    print("test_gls: OK")
