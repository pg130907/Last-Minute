"""Module 1 gate — LCC extraction drops isolated nodes; LCC is fully connected.

Also covers the acceptance sub-checks:
- load_8node()'s distance matrix matches the spec §3.3 fixture,
- update_weights() changes tau on vol change (BPR is live).
"""
from __future__ import annotations

import sys

import networkx as nx
import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from engine_numpy.graph import BIG_FINITE, extract_lcc  # noqa: E402
from engine_numpy.loaders import (  # noqa: E402
    _D8,
    _Q8,
    load_8node,
    make_disconnected_synthetic,
)


def test_load_8node_matches_spec_matrix() -> None:
    g = load_8node()
    assert g.n == 9, f"expected 9 nodes (depot+8 customers), got {g.n}"
    # Off-diagonal distances must match spec verbatim.
    d = np.asarray(g.dist)
    for i in range(9):
        for j in range(9):
            if i == j:
                assert d[i, j] == 0
            else:
                assert abs(d[i, j] - _D8[i, j]) < 1e-4, f"dist[{i},{j}] mismatch"
    # Demand vector and capacity.
    assert np.allclose(g.demand, _Q8)
    assert g.capacity == 8.0
    # BPR trivially: cap large, vol 0 → tau == tau0
    assert np.allclose(g.tau, g.tau0)


def test_update_weights_changes_tau_on_vol() -> None:
    g = load_8node()
    tau_before = g.tau.copy()
    # Slam a heavy background load on edge 0→4, and give that edge finite cap.
    g.cap[0, 4] = 100.0
    g.vol[0, 4] = 200.0    # v/c = 2 → 0.15*(2^4) = 2.4× tau0
    g.update_weights(t=0.0)
    assert g.tau[0, 4] > tau_before[0, 4] * 3.3, (
        f"BPR did not raise tau: before={tau_before[0,4]}, after={g.tau[0,4]}"
    )
    # Untouched edge stays put.
    assert abs(g.tau[1, 2] - tau_before[1, 2]) < 1e-4


def test_lcc_drops_isolated_nodes() -> None:
    g = make_disconnected_synthetic(main_n=6, isolated_n=2, seed=7)
    assert g.n == 8
    kept_graph, kept = extract_lcc(g, strongly=True)
    assert kept_graph.n == 6, f"expected 6 nodes after LCC, got {kept_graph.n}"
    assert set(kept.tolist()) == set(range(6))
    # LCC must be strongly connected.
    G = nx.from_numpy_array(kept_graph.adj_mask.astype(np.uint8), create_using=nx.DiGraph)
    assert nx.is_strongly_connected(G), "LCC output is not strongly connected"


def test_lcc_produces_no_inf_inside_component() -> None:
    """Once we're inside the LCC, tau0 must be finite on every real edge."""
    g = make_disconnected_synthetic(main_n=6, isolated_n=2, seed=11)
    kept_graph, _ = extract_lcc(g)
    # For every edge marked present, tau0 is finite.
    edges = kept_graph.adj_mask
    assert np.isfinite(kept_graph.tau0[edges]).all()
    assert np.isfinite(kept_graph.dist[edges]).all()
    # Non-edges' emis is BIG_FINITE (finite-guarded), never inf/NaN.
    assert np.isfinite(kept_graph.emis).all()
    assert (kept_graph.emis[~edges & ~np.eye(kept_graph.n, dtype=bool)] == BIG_FINITE).all()


def test_lcc_noop_when_already_connected() -> None:
    g = load_8node()  # complete graph → already one component
    kept_graph, kept = extract_lcc(g)
    assert kept_graph is g
    assert len(kept) == g.n


if __name__ == "__main__":
    test_load_8node_matches_spec_matrix()
    test_update_weights_changes_tau_on_vol()
    test_lcc_drops_isolated_nodes()
    test_lcc_produces_no_inf_inside_component()
    test_lcc_noop_when_already_connected()
    print("Module 1 (graph + loaders + LCC): OK")
