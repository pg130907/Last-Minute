"""Module 5b gate — FIFO tau lookup.

Invariant: for any edge (i,j) and any two departure times t1 < t2, arrivals
must satisfy arrive(t1) ≤ arrive(t2). I.e., leaving later never lands you
earlier. This is the classic TDVRP FIFO property.

Also checks:
  - event shocks raise tau on the affected slice
  - forecast/arima returns a plausible horizon
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from data.traffic.synth_generator import generate_series  # noqa: E402
from engine_numpy.graph import BIG_FINITE, Graph, build_from_dist  # noqa: E402
from engine_numpy.loaders import make_synthetic  # noqa: E402
from engine_numpy.predictor import (  # noqa: E402
    EventShock,
    TrafficSchedule,
    arima_forecast,
    make_tau_lookup,
)


def _demo_graph(seed: int = 3) -> Graph:
    g = make_synthetic(n=8, seed=seed, capacity=100.0)
    # Give edges a realistic capacity so BPR bites when we crank volume up.
    g.cap = np.where(g.adj_mask, 300.0, 0.0).astype(np.float32)
    g.update_weights()
    return g


def _build_synthetic_schedule(g: Graph, bucket_min: int = 30, seed: int = 0) -> TrafficSchedule:
    """Turn a per-edge synthetic 24h series into a (S, N, N) volume schedule."""
    n_edges_flat = g.n * g.n
    series = generate_series(n_edges=n_edges_flat, seed=seed,
                             slot_min=bucket_min, base_vol=150.0, peak_vol=1000.0)
    vol_series = series.volumes.reshape(-1, g.n, g.n).astype(np.float32)
    # Zero out non-edges.
    for s in range(vol_series.shape[0]):
        vol_series[s] = np.where(g.adj_mask, vol_series[s], 0.0)
    return TrafficSchedule.from_bpr(g, vol_series, bucket_min=bucket_min)


def test_fifo_leave_later_arrive_no_earlier() -> None:
    g = _demo_graph()
    sched = _build_synthetic_schedule(g, bucket_min=30, seed=7)
    lookup = make_tau_lookup(sched)

    # Sweep a fine grid of departure times over the day; check arrivals monotone.
    t_grid = np.linspace(0, 24 * 60 - 1, 400)
    violations = 0
    checked_pairs = 0
    for i in range(g.n):
        for j in range(g.n):
            if not g.adj_mask[i, j]:
                continue
            arrivals = np.array([t + lookup(g, i, j, t) for t in t_grid])
            # Must be monotone non-decreasing.
            deltas = np.diff(arrivals)
            if (deltas < -1e-4).any():
                violations += int((deltas < -1e-4).sum())
                bad_idx = int(np.argmin(deltas))
                print(
                    f"  FIFO break on edge {i}→{j}: "
                    f"t1={t_grid[bad_idx]:.1f} arr1={arrivals[bad_idx]:.3f}, "
                    f"t2={t_grid[bad_idx+1]:.1f} arr2={arrivals[bad_idx+1]:.3f}"
                )
            checked_pairs += 1
    print(f"  ├─ checked {checked_pairs} edges × 400 depart samples")
    assert violations == 0, f"FIFO violated {violations} times"


def test_event_shock_raises_tau_on_active_slice() -> None:
    g = _demo_graph()
    sched = _build_synthetic_schedule(g, bucket_min=30, seed=11)
    lookup = make_tau_lookup(sched)

    # Pick two edges to shock at 15:00–16:00.
    edge_pairs = [(0, 1), (2, 3)]
    for (i, j) in edge_pairs:
        assert g.adj_mask[i, j], "test fixture picked a non-edge"

    tau_before = np.array([lookup(g, i, j, 15 * 60 + 10) for (i, j) in edge_pairs])
    ev = EventShock(name="school_dismissal", edges=edge_pairs,
                    start_min=15 * 60, end_min=16 * 60, extra_vol=2000.0)
    sched.apply_events(g, [ev])
    tau_after = np.array([lookup(g, i, j, 15 * 60 + 10) for (i, j) in edge_pairs])
    print(f"  ├─ tau at 15:10 before: {tau_before}")
    print(f"  ├─ tau at 15:10 after : {tau_after}")
    assert (tau_after > tau_before + 1e-3).all(), (
        f"event shock did not raise tau: before={tau_before} after={tau_after}"
    )

    # After the event ends, tau should snap back on non-shocked slices…
    # but FIFO envelope keeps arrival monotone so post-event tau at 17:00 is
    # bounded by the peak during the shock. Not a strict equality test.
    tau_post = np.array([lookup(g, i, j, 17 * 60) for (i, j) in edge_pairs])
    print(f"  └─ tau at 17:00 after : {tau_post}  (may reflect FIFO envelope)")


def test_arima_forecast_shape_and_positivity() -> None:
    series = generate_series(n_edges=1, seed=0).volumes[0]
    fcst = arima_forecast(series[:80], horizon=16, order=(2, 0, 1))
    assert fcst.shape == (16,)
    assert (fcst >= 0).all(), "forecast should be non-negative volumes"
    # Not asserting closeness to true (bimodal daily shape needs SARIMA),
    # just: it returns something plausible and finite.
    assert np.isfinite(fcst).all()


if __name__ == "__main__":
    test_fifo_leave_later_arrive_no_earlier()
    test_event_shock_raises_tau_on_active_slice()
    test_arima_forecast_shape_and_positivity()
    print("test_fifo: OK")
